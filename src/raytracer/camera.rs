use rand::Rng;
use std::{cell::RefCell, fs::File, io::Write};

use indicatif::ProgressBar;

use crate::{
    Result,
    raytracer::{
        color::Color,
        hitable::Hitable,
        options::RenderOptions,
        ray::Ray,
        vec3::{Point3, Vec3},
    },
};

pub struct Camera {
    pub origin: Point3,
    pub render_options: RenderOptions,
    rng: RefCell<rand::rngs::ThreadRng>,
}

impl Camera {
    pub fn new(origin: Point3, render_options: RenderOptions) -> Self {
        Camera {
            origin,
            render_options,
            rng: RefCell::new(rand::rng()),
        }
    }

    pub fn render<T: Hitable>(&self, world: &T) -> Result<()> {
        // Set up the progress bar
        let progress =
            ProgressBar::new((self.render_options.width * self.render_options.height) as u64);

        // Open the output file
        let mut file = File::create(&self.render_options.file_name)?;

        // Calculate the location of the upper left pixel.
        let viewport_u = Vec3::new(self.render_options.viewport_width(), 0.0, 0.0);
        let viewport_v = Vec3::new(0.0, -self.render_options.viewport_height, 0.0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        let pixel_delta_u = viewport_u / self.render_options.width as f32;
        let pixel_delta_v = viewport_v / self.render_options.height as f32;

        // Calculate the location of the upper left pixel.
        let viewport_upper_left = self.origin
            - Vec3::new(0.0, 0.0, self.render_options.focal_length)
            - viewport_u / 2.0
            - viewport_v / 2.0;
        let pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

        // Render
        let image_width = self.render_options.width;
        let image_height = self.render_options.height;

        writeln!(file, "P3\n{} {}\n255\n", image_width, image_height)?;

        for j in 0..image_height {
            for i in 0..image_width {
                let mut pixel_color = Color::black();
                for _ in 0..self.render_options.samples_per_pixel {
                    // Calculate the pixel sample location.
                    let offset = self.sample_square();
                    let pixel_sample = pixel00_loc
                        + (pixel_delta_u * (i as f32 + offset.x))
                        + (pixel_delta_v * (j as f32 + offset.y));
                    let ray_direction = pixel_sample - self.origin;
                    let ray = Ray::new(self.origin, ray_direction);

                    pixel_color += ray.color(world);
                }
                writeln!(
                    file,
                    "{}",
                    pixel_color / self.render_options.samples_per_pixel as f32
                )?;
                progress.inc(1);
            }
        }
        progress.finish();
        Ok(())
    }

    fn sample_square(&self) -> Vec3 {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        let mut rng = self.rng.borrow_mut();
        let x = rng.random::<f32>() - 0.5;
        let y = rng.random::<f32>() - 0.5;
        Vec3::new(x, y, 0.0)
    }
}
