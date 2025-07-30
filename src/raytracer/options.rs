use clap::{Args, Parser};

/// Rendering options for the ray tracer.
#[derive(Debug, Args)]
pub struct RenderOptions {
    /// Number of rows in the concentration table
    #[arg(short = 'W', long = "width", default_value_t = 960)]
    pub width: usize,

    /// Number of columns in the concentration table
    #[arg(short = 'H', long = "height", default_value_t = 540)]
    pub height: usize,

    /// Output file name
    #[arg(short = 'o', long = "output", default_value = "image.ppm")]
    pub file_name: String,
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
