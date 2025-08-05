#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#[cfg(target_os = "cuda")]
use cuda_std::prelude::*;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

use crate::vec3::{Point3, Real, Vec3};
use crate::{color::Color, hitable::HitKind};
use grid_nd::GridND;

#[cfg(not(target_os = "cuda"))]
use indicatif::ProgressBar;

pub mod color;
pub mod hitable;
pub mod hitable_list;
pub mod materials;
pub mod ray;
pub mod sphere;
pub mod vec3;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct ImageRenderOptions {
    pub samples_per_pixel: usize,
    pub origin: Point3,
    pub max_depth: usize,

    pub defocus_angle: Real,
    pub defocus_disk_u: Vec3,
    pub defocus_disk_v: Vec3,

    pub pixel00_loc: Point3,
    pub pixel_delta_u: Vec3,
    pub pixel_delta_v: Vec3,
}

#[cfg(not(target_os = "cuda"))]
pub fn render_image(grid: &mut GridND<Color, 2>, world: &HitKind, options: &ImageRenderOptions) {
    // Set up the progress bar
    let progress = ProgressBar::new(grid.shape().iter().product::<usize>() as u64);
    for i in 0..grid.shape()[1] {
        for j in 0..grid.shape()[0] {
            let mut pixel_color = Color::black();
            for _ in 0..options.samples_per_pixel {
                // Calculate the pixel sample location.

                use crate::ray::Ray;
                let offset = Vec3::sample_square();
                let pixel_sample = options.pixel00_loc
                    + (options.pixel_delta_u * (i as Real + offset.x))
                    + (options.pixel_delta_v * (j as Real + offset.y));

                // Apply defocus if enabled
                let ray_origin = options.origin
                    + if options.defocus_angle > 0.0 {
                        let offset = Vec3::random_in_unit_disk();
                        options.defocus_disk_u * offset.x + options.defocus_disk_v * offset.y
                    } else {
                        Vec3::zero()
                    };

                let ray_direction = pixel_sample - ray_origin;
                let ray = Ray::new(ray_origin, ray_direction);

                pixel_color += ray.color(options.max_depth, world);
            }
            // Store the pixel color in the grid
            *grid.at_mut(j).at_mut(i) = pixel_color / options.samples_per_pixel as Real;
        }
        progress.inc(grid.shape()[0] as u64);
    }
    progress.finish();
}
