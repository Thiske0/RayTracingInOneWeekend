use clap::{Args, Parser};

/// Rendering options for the ray tracer.
#[derive(Debug, Args)]
pub struct RenderOptions {
    /// Width of the image
    #[arg(short = 'W', long = "width", default_value_t = 960)]
    pub width: usize,

    /// Height of the image
    #[arg(short = 'H', long = "height", default_value_t = 540)]
    pub height: usize,

    /// Height of the viewport
    #[arg(short = 'v', long = "viewport-height", default_value_t = 2.0)]
    pub viewport_height: f32,

    /// Focal length of the camera
    #[arg(short = 'f', long = "focal-length", default_value_t = 1.0)]
    pub focal_length: f32,

    /// Output file name
    #[arg(short = 'o', long = "output", default_value = "image.ppm")]
    pub file_name: String,
}

impl RenderOptions {
    /// Returns the aspect ratio of the image
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    /// Returns the viewport width based on the aspect ratio and height
    pub fn viewport_width(&self) -> f32 {
        self.viewport_height * self.aspect_ratio()
    }
}

/// Ray Tracing in One Weekend
///
/// This program implements a simple ray tracer in Rust.
#[derive(Debug, Parser)]
#[command(version)]
pub struct Options {
    #[command(flatten)]
    pub render: RenderOptions,
}
