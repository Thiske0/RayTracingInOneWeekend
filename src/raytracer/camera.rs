use std::{fs::File, io::Write};

use indicatif::ProgressBar;

use crate::{
    Result,
    raytracer::{
        color::Color,
        hitable::Hitable,
        options::RenderOptions,
        ray::Ray,
        vec3::{Point3, Real, Vec3},
    },
};

pub struct Camera {
    pub render_options: RenderOptions,
}

impl Camera {
    pub fn new(render_options: RenderOptions) -> Self {
        Camera { render_options }
    }

    fn initilize(&self) -> (Point3, Vec3, Vec3, Vec3, Vec3, Vec3) {
        let origin = self.render_options.lookfrom;

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        let w = (origin - self.render_options.lookat).normalize();
        let u = self.render_options.vup.cross(w).normalize();
        let v = w.cross(u);

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

        (
            origin,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
            defocus_disk_u,
            defocus_disk_v,
        )
    }

    pub fn render<T: Hitable>(&self, world: &T) -> Result<()> {
        // Set up the progress bar
        let progress =
            ProgressBar::new((self.render_options.width * self.render_options.height) as u64);

        // Open the output file
        let mut file = File::create(&self.render_options.file_name)?;

        // Initialize camera parameters
        let (origin, pixel00_loc, pixel_delta_u, pixel_delta_v, defocus_disk_u, defocus_disk_v) =
            self.initilize();

        // Render
        let image_width = self.render_options.width;
        let image_height = self.render_options.height;

        writeln!(file, "P3\n{} {}\n255\n", image_width, image_height)?;

        for j in 0..image_height {
            for i in 0..image_width {
                let mut pixel_color = Color::black();
                for _ in 0..self.render_options.samples_per_pixel {
                    // Calculate the pixel sample location.
                    let offset = Vec3::sample_square();
                    let pixel_sample = pixel00_loc
                        + (pixel_delta_u * (i as Real + offset.x))
                        + (pixel_delta_v * (j as Real + offset.y));

                    // Apply defocus if enabled
                    let ray_origin = origin
                        + if self.render_options.defocus_angle > 0.0 {
                            Camera::defocus_disk_sample(defocus_disk_u, defocus_disk_v)
                        } else {
                            Vec3::zero()
                        };

                    let ray_direction = pixel_sample - ray_origin;
                    let ray = Ray::new(ray_origin, ray_direction);

                    pixel_color += ray.color(self.render_options.max_depth, world);
                }
                writeln!(
                    file,
                    "{}",
                    pixel_color / self.render_options.samples_per_pixel as Real
                )?;
                progress.inc(1);
            }
        }
        progress.finish();
        Ok(())
    }

    fn defocus_disk_sample(defocus_disk_u: Vec3, defocus_disk_v: Vec3) -> Vec3 {
        let offset = Vec3::random_in_unit_disk();
        defocus_disk_u * offset.x + defocus_disk_v * offset.y
    }
}
