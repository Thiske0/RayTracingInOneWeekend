use cust::{
    context::{legacy::CurrentContext, Context}, device::Device, error::CudaResult, launch, memory::{AsyncCopyDestination, DeviceBox, DeviceBuffer}, module::Module, stream::{Stream, StreamFlags}
};
use gpu_builder::NiceBuilder;
use gpu_rand::DefaultRand;
use grid_nd::GridND;
use image::{ImageBuffer, RgbImage};
use rand::Rng;
use simple_ray_tracer_kernels::{ImageRenderOptions, hitables::HitKind};
use std::{fs, path::Path};

use crate::{Result, raytracer::options::RenderOptions};

use simple_ray_tracer_kernels::{color::Color, vec3::Real};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub struct Camera {
    pub render_options: RenderOptions,
}

impl Camera {
    pub fn new(render_options: RenderOptions) -> Self {
        Camera { render_options }
    }

    unsafe fn initilize(
        &self,
        contexts: &[(Context, Module, Stream)],
    ) -> Result<(ImageRenderOptions, Vec<DeviceBuffer<DefaultRand>>)> {
        let origin = self.render_options.lookfrom;

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        let w = (origin - self.render_options.lookat).normalize();
        let u = self.render_options.vup.cross(&w).normalize();
        let v = w.cross(&u);

        // Calculate the location of the upper left pixel.
        let viewport_u = u * self.render_options.viewport_width();
        let viewport_v = v * -self.render_options.viewport_height();

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        let pixel_delta_u = viewport_u / self.render_options.width as Real;
        let pixel_delta_v = viewport_v / self.render_options.height as Real;

        // Calculate the location of the upper left pixel.
        let viewport_upper_left =
            origin - w * self.render_options.focus_distance - viewport_u / 2.0 - viewport_v / 2.0;
        let pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

        // Calculate the camera defocus disk basis vectors.
        let defocus_radius = self.render_options.focus_distance
            * (self.render_options.defocus_angle / 2.0).to_radians().tan();
        let defocus_disk_u = u * defocus_radius;
        let defocus_disk_v = v * defocus_radius;

        let seed = rand::rng().random();
        let total_elems = self.render_options.width * self.render_options.height;
        let rand_states = DefaultRand::initialize_states(seed, total_elems);
        let rand_states = contexts.into_iter().map(|(ctx, _module, stream)| {
            CurrentContext::set_current(ctx)?;
            unsafe {
                let mut device_buffer = DeviceBuffer::uninitialized_async(total_elems, stream)?;
                device_buffer.async_copy_from(rand_states.as_slice(), stream)?;
                Ok(device_buffer)
            }
        }).collect::<Result<Vec<_>>>()?;

        Ok((
            ImageRenderOptions {
                samples_per_pixel: self.render_options.samples_per_pixel,
                origin,
                max_depth: self.render_options.max_depth,
                background: self.render_options.background,
                defocus_angle: self.render_options.defocus_angle,
                defocus_disk_u,
                defocus_disk_v,
                pixel00_loc,
                pixel_delta_u,
                pixel_delta_v,
            },
            rand_states,
        ))
    }

    pub fn render<'a>(&self, world: HitKind<'a>) -> Result<()> {
        // Render
        let image_width = self.render_options.width;
        let image_height = self.render_options.height;

        let image_grid = GridND::new([image_height, image_width], Color::black());

        cust::init(cust::CudaFlags::empty())?;
        let num_gpus = cust::device::Device::num_devices()? as usize;

        let mut contexts = Vec::with_capacity(num_gpus);
        for i in 0..num_gpus {
            let device = Device::get_device(i as u32)?;
            let ctx = Context::new(device)?;
            let module = Module::from_ptx(PTX, &[])?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            contexts.push((ctx, module, stream));
        }

        //TODO: get rid of clone by changing lifetimes
        let binding = world.clone();
        let lights = binding.get_lights();

        let world_copy = world.clone();
        let lights_copy = lights.clone();

        let callback = |image_grid: &GridND<Color, 2>| {
            if self.render_options.calculate_noise {
                let noise_callback = |other_image_grid: &GridND<Color, 2>| -> Result<()> {
                    let mut total_variance = 0.0;
                    for (c1, c2) in (&image_grid)
                        .into_iter()
                        .zip((&other_image_grid).into_iter())
                    {
                        for (p1, p2) in c1.into_iter().zip(c2.into_iter()) {
                            total_variance += p1.variance(p2);
                        }
                    }
                    let noise = (total_variance / (2 * image_width * image_height) as Real).sqrt();
                    println!("Estimated noise: {}", noise);
                    Ok(())
                };
                self.render_with_callback(
                    image_grid.clone(),
                    world_copy,
                    lights_copy,
                    &contexts,
                    noise_callback,
                )?;
            }
            self.save_image(image_grid)
        };
        self.render_with_callback(image_grid, world, lights, &contexts, callback)?;

        for (ctx, module, stream) in contexts.into_iter().rev() {
            // Drop streams and modules explicitly first if needed
            drop(stream);
            drop(module);
            drop(ctx); // drop context last
        }
        Ok(())
    }

    fn render_with_callback(
        &self,
        mut image_grid: GridND<Color, 2>,
        world: HitKind,
        lights: HitKind,
        contexts: &[(Context, Module, Stream)],
        callback: impl FnOnce(&GridND<Color, 2>) -> Result<()>,
    ) -> Result<()> {
        unsafe {
            // Initialize camera parameters
            let (image_render_options, rand_states_device) = self.initilize(contexts)?;

            if self.render_options.gpu_render {
                Self::render_gpu(
                    image_grid,
                    world,
                    lights,
                    &image_render_options,
                    rand_states_device,
                    contexts,
                    callback,
                )?;
            } else {
                Self::render_cpu(
                    &mut image_grid,
                    world,
                    lights,
                    &image_render_options,
                    callback,
                )?;
            }
        }
        Ok(())
    }

    fn save_image(&self, image_grid: &GridND<Color, 2>) -> Result<()> {
        let img: RgbImage = ImageBuffer::from_fn(
            image_grid.shape()[1] as u32,
            image_grid.shape()[0] as u32,
            |x, y| {
                let color = *image_grid.at(y as usize).at(x as usize);
                color.into()
            },
        );

        let path = Path::new(&self.render_options.file_name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        img.save(&self.render_options.file_name)?;
        Ok(())
    }

    #[allow(unused)]
    unsafe fn render_gpu<'a>(
        image_grid: GridND<Color, 2>,
        world: HitKind<'a>,
        lights: HitKind<'a>,
        image_render_options: &ImageRenderOptions,
        rand_states_device: Vec<DeviceBuffer<DefaultRand>>,
        contexts: &[(Context, Module, Stream)],
        callback: impl FnOnce(&GridND<Color, 2>) -> Result<()>,
    ) -> Result<()> {
        unsafe {
            let num_gpus = contexts.len();


            // move data to GPUs
            let mut device_data = contexts.iter().zip(rand_states_device).enumerate().map(|(i, ((ctx, module, stream), rand_states))| {
                CurrentContext::set_current(ctx)?;
                let render_image = module.get_function("render_image")?;

            
                let (_, recommended_block_size) =
                    render_image.suggested_launch_configuration(0, 0.into())?;
                let mut local_grid_shape = image_grid.shape();
                local_grid_shape[0] = (local_grid_shape[0]+num_gpus-1)/num_gpus;
                let (blocks, threads) =
                    GridND::<Color, 2>::grid_and_block_size(local_grid_shape, recommended_block_size);
                
                let world_device = world.clone().build_device(stream)?;
                let lights_device = lights.clone().build_device(stream)?;
                let image_render_options_device = DeviceBox::new_async(image_render_options, stream)?;
                
                let local_image_grid = GridND::new(local_grid_shape, Color::black());
                let mut image_grid_device = local_image_grid.build_device(stream)?;

                let ofset_device = DeviceBox::new_async(&(i as u32), stream)?;
                let num_streams_device = DeviceBox::new_async(&(num_gpus as u32), stream)?;

                Ok((render_image, blocks, threads, world_device, lights_device, image_render_options_device, image_grid_device, rand_states, ofset_device, num_streams_device))
            }).collect::<CudaResult<Vec<_>>>()?;


            contexts.iter().zip(&device_data).enumerate().map(|(i, ((ctx, module, stream), device_data))| {
                CurrentContext::set_current(ctx)?;
                let (render_image, blocks, threads, world_device, lights_device, image_render_options_device, image_grid_device, rand_states, ofset_device, num_streams_device) = device_data;

                launch!(
                    render_image<<<blocks, threads, 0, stream>>>(
                        image_grid_device.as_device_ptr()?.as_mut_ptr(),
                        world_device.as_device_ptr()?,
                        image_render_options_device.as_device_ptr(),
                        lights_device.as_device_ptr()?,
                        rand_states.as_device_ptr().as_mut_ptr(),
                        ofset_device.as_device_ptr(),
                        num_streams_device.as_device_ptr()
                    )
                )?;
                Ok(())
            }).collect::<CudaResult<Vec<_>>>()?;

            // drop device data to free GPU memory, but keep image grids alive for copying back results
            let mut image_grids_device = contexts.iter().zip(device_data).enumerate().map(|(i, ((ctx, module, stream), device_data))| {
                CurrentContext::set_current(ctx)?;
                let (_render_image, _blocks, _threads, world_device, lights_device, image_render_options_device, image_grid_device, rand_states, ofset_device, num_streams_device) = device_data;
                // world_device.drop_async(stream)?;
                // lights_device.drop_async(stream)?;
                image_render_options_device.drop_async(stream)?;
                rand_states.drop_async(stream)?;
                ofset_device.drop_async(stream)?;
                num_streams_device.drop_async(stream)?;
                Ok(image_grid_device)
            }).collect::<CudaResult<Vec<_>>>()?;


            // synchronize
            contexts.iter().map(|(ctx, _module, stream)| {
                CurrentContext::set_current(ctx)?;
                stream.synchronize()?;
                Ok(())
            }).collect::<CudaResult<Vec<_>>>()?;

            // gather results
            let result_grids = image_grids_device.iter_mut().zip(contexts).map(|(grid, (ctx, _module, _stream))| {
                CurrentContext::set_current(ctx)?;
                grid.copy_back()
            }).collect::<CudaResult<Vec<_>>>()?;

            let mut result_grid = image_grid;
            let mut current_stream = 0;
            let mut local_row_index = 0;
            for mut result_grid_row in (&mut result_grid).into_iter() {
                let local_row = result_grids[current_stream].at(local_row_index);
                for (result_pixel, local_pixel) in (&mut result_grid_row).into_iter().zip(local_row.into_iter()) {
                    *result_pixel = *local_pixel;
                }
                current_stream += 1;
                if(current_stream == num_gpus) {
                    current_stream = 0;
                    local_row_index += 1;
                }
            }

            callback(&result_grid)?;

            Ok(())
        }
    }

    #[allow(unused)]
    fn render_cpu<'a>(
        image_grid: &mut GridND<Color, 2>,
        world: HitKind<'a>,
        lights: HitKind<'a>,
        image_render_options: &ImageRenderOptions,
        callback: impl FnOnce(&GridND<Color, 2>) -> Result<()>,
    ) -> Result<()> {
        simple_ray_tracer_kernels::render_image(image_grid, &world, image_render_options, &lights);
        callback(image_grid)?;
        Ok(())
    }
}
