use std::{fs::File, io::Write};

use clap::Parser;
use indicatif::ProgressBar;

use simple_ray_tracer::{
    Result,
    raytracer::{
        options::Options,
        ray::Ray,
        vec3::{Point3, Vec3},
    },
};

fn main() -> Result<()> {
    // Parse command line options
    let options = Options::parse();

    // Set up the progress bar
    let progress = ProgressBar::new((options.render.width * options.render.height) as u64);

    // Open the output file
    let mut file = File::create(&options.render.file_name)?;

    // Camera setup
    let camera_origin = Point3::new(0.0, 0.0, 0.0);

    // Calculate the location of the upper left pixel.
    let viewport_u = Vec3::new(options.render.viewport_width(), 0.0, 0.0);
    let viewport_v = Vec3::new(0.0, -options.render.viewport_height, 0.0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    let pixel_delta_u = viewport_u / options.render.width as f32;
    let pixel_delta_v = viewport_v / options.render.height as f32;

    // Calculate the location of the upper left pixel.
    let viewport_upper_left = camera_origin
        - Vec3::new(0.0, 0.0, options.render.focal_length)
        - viewport_u / 2.0
        - viewport_v / 2.0;
    let pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

    // Render
    let image_width = options.render.width;
    let image_height = options.render.height;

    writeln!(file, "P3\n{} {}\n255\n", image_width, image_height)?;

    for j in 0..image_height {
        for i in 0..image_width {
            let pixel_center =
                pixel00_loc + (pixel_delta_u * i as f32) + (pixel_delta_v * j as f32);
            let ray_direction = pixel_center - camera_origin;
            let ray = Ray::new(camera_origin, ray_direction);

            let pixel_color = ray.color();

            writeln!(file, "{}", pixel_color)?;
            progress.inc(1);
        }
    }
    progress.finish();
    Ok(())
}
