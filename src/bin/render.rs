use std::{fs::File, io::Write};

use clap::Parser;
use indicatif::ProgressBar;

use simple_ray_tracer::{
    Result,
    raytracer::{color::Color, options::Options},
};

fn main() -> Result<()> {
    // Parse command line options
    let options = Options::parse();

    // Set up the progress bar
    let progress = ProgressBar::new((options.render.width * options.render.height) as u64);

    // Open the output file
    let mut file = File::create(options.render.file_name)?;

    // Render
    let image_width = options.render.width;
    let image_height = options.render.height;

    writeln!(file, "P3\n{} {}\n255\n", image_width, image_height)?;

    for j in 0..image_height {
        for i in 0..image_width {
            let v = Color::new(
                i as f32 / (image_width - 1) as f32,
                j as f32 / (image_height - 1) as f32,
                0.0,
            );

            writeln!(file, "{}", v)?;
            progress.inc(1);
        }
    }
    progress.finish();
    Ok(())
}
