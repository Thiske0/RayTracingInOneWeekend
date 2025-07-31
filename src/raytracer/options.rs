use std::str::FromStr;

use clap::{Args, Parser};

use crate::raytracer::vec3::{Point3, Real, Vec3};

/// Rendering options for the ray tracer.
#[derive(Debug, Args)]
pub struct RenderOptions {
    /// Width of the image
    #[arg(short = 'W', long = "width", default_value_t = 960)]
    pub width: usize,

    /// Height of the image
    #[arg(short = 'H', long = "height", default_value_t = 540)]
    pub height: usize,

    /// Vertical field of view in degrees
    #[arg(short = 'v', long = "vertical-fov", default_value_t = 70.0)]
    pub vertical_fov: Real,

    /// Focal length of the camera
    #[arg(short = 'f', long = "focal-length", default_value_t = 1.0)]
    pub focal_length: Real,

    /// Number of samples per pixel
    #[arg(short = 's', long = "samples", default_value_t = 10)]
    pub samples_per_pixel: usize,

    /// Maximum depth of ray bounces
    #[arg(short = 'd', long = "max-depth", default_value_t = 10)]
    pub max_depth: usize,

    #[arg(long = "look-from", default_value = "0,0,0")]
    pub lookfrom: Point3,
    #[arg(long = "look-at", default_value = "0,0,-1")]
    pub lookat: Point3,
    #[arg(long = "vup", default_value = "0,1,0")]
    pub vup: Vec3,

    /// Output file name
    #[arg(short = 'o', long = "output", default_value = "image.ppm")]
    pub file_name: String,
}

impl FromStr for Vec3 {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err("Expected three comma-separated values".to_string());
        }
        let x = parts[0]
            .parse()
            .map_err(|_| "Invalid x value".to_string())?;
        let y = parts[1]
            .parse()
            .map_err(|_| "Invalid y value".to_string())?;
        let z = parts[2]
            .parse()
            .map_err(|_| "Invalid z value".to_string())?;
        Ok(Vec3::new(x, y, z))
    }
}

impl RenderOptions {
    /// Returns the aspect ratio of the image
    pub fn aspect_ratio(&self) -> Real {
        self.width as Real / self.height as Real
    }

    /// Returns the viewport height based on the fov
    pub fn viewport_height(&self) -> Real {
        let h = (self.vertical_fov.to_radians() / 2.0).tan() * self.focal_length;
        2.0 * h * self.focal_length
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
