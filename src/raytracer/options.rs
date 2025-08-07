use clap::{Args, Parser};

use simple_ray_tracer_kernels::vec3::{Point3, Real, Vec3};

/// Rendering options for the ray tracer.
#[derive(Debug, Args)]
pub struct RenderOptions {
    /// Width of the image
    #[arg(short = 'W', long = "width", default_value_t = 1920)]
    pub width: usize,

    /// Height of the image
    #[arg(short = 'H', long = "height", default_value_t = 1080)]
    pub height: usize,

    /// Vertical field of view in degrees
    #[arg(short = 'v', long = "vertical-fov", default_value_t = 20.0)]
    pub vertical_fov: Real,

    /// Number of samples per pixel
    #[arg(short = 's', long = "samples", default_value_t = 20)]
    pub samples_per_pixel: usize,

    /// Maximum depth of ray bounces
    #[arg(short = 'd', long = "max-depth", default_value_t = 50)]
    pub max_depth: usize,

    #[arg(long = "look-from", default_value = "13,2,3")]
    pub lookfrom: Point3,
    #[arg(long = "look-at", default_value = "0,0,0")]
    pub lookat: Point3,
    #[arg(long = "vup", default_value = "0,1,0")]
    pub vup: Vec3,

    /// depth of field
    #[arg(short = 'f', long = "focus-distance", default_value_t = 10.0)]
    pub focus_distance: Real,
    #[arg(short = 'a', long = "defocus-angle", default_value_t = 0.6)]
    pub defocus_angle: Real,

    /// Output file name
    #[arg(short = 'o', long = "output", default_value = "image.png")]
    pub file_name: String,
}

impl RenderOptions {
    /// Returns the aspect ratio of the image
    pub fn aspect_ratio(&self) -> Real {
        self.width as Real / self.height as Real
    }

    /// Returns the viewport height based on the fov
    pub fn viewport_height(&self) -> Real {
        let h = (self.vertical_fov.to_radians() / 2.0).tan();
        2.0 * h * self.focus_distance
    }

    /// Returns the viewport width based on the aspect ratio and height
    pub fn viewport_width(&self) -> Real {
        self.viewport_height() * self.aspect_ratio()
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
