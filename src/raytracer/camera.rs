use cust::{
    memory::{AsyncCopyDestination, DeviceBox, DeviceBuffer},
    module::Module,
    stream::{Stream, StreamFlags},
};
use gpu_rand::DefaultRand;
use grid_nd::GridND;
use image::{ImageBuffer, RgbImage};
use rand::Rng;
use simple_ray_tracer_kernels::{ImageRenderOptions, hitable::HitKind};

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
        stream: &Stream,
    ) -> Result<(ImageRenderOptions, DeviceBuffer<DefaultRand>)> {
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
        let rand_states = unsafe {
            let mut device_buffer = DeviceBuffer::uninitialized_async(total_elems, stream)?;
            device_buffer.async_copy_from(rand_states.as_slice(), stream)?;
            device_buffer
        };

        Ok((
            ImageRenderOptions {
                samples_per_pixel: self.render_options.samples_per_pixel,
                origin,
                max_depth: self.render_options.max_depth,
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

    pub fn render(&self, world: &HitKind) -> Result<()> {
        // Render
        let image_width = self.render_options.width;
        let image_height = self.render_options.height;

        let mut grid = GridND::new([image_height, image_width], Color::black());

        let _ctx = cust::quick_init()?;
        let module = Module::from_ptx(PTX, &[])?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        unsafe {
            // Initialize camera parameters
            let (image_render_options, rand_states_device) = self.initilize(&stream)?;

            let world_device = DeviceBox::new_async(world, &stream)?;
            let image_render_options_device = DeviceBox::new_async(&image_render_options, &stream)?;
            let grid_device = grid.to_device_async(&stream)?;

            /*launch!(
                module.render<<<blocks, threads, 0, stream>>>(
                    self.buffers.accumulated_buffer.as_device_ptr(),
                    self.buffers.viewport,
                    scene.as_device_ptr(),
                    self.buffers.rand_states.as_unified_ptr()
                )
            )?;*/

            world_device.drop_async(&stream)?;
            image_render_options_device.drop_async(&stream)?;
            rand_states_device.drop_async(&stream)?;

            stream.synchronize()?;

            grid.copy_back(&grid_device)?;

            //CPU rendering
            simple_ray_tracer_kernels::render_image(&mut grid, world, &image_render_options);
        }

        let img: RgbImage =
            ImageBuffer::from_fn(image_width as u32, image_height as u32, |x, y| {
                let color = *grid.at(y as usize).at(x as usize);
                color.into()
            });
        img.save(&self.render_options.file_name)?;
        Ok(())
    }
}
