#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#[cfg(target_os = "cuda")]
use cuda_std::glam::{UVec2, UVec3};
#[cfg(target_os = "cuda")]
use cuda_std::prelude::*;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
#[cfg(target_os = "cuda")]
use gpu_rand::DefaultRand;

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

#[cfg(target_os = "cuda")]
#[kernel]
#[cfg(target_os = "cuda")]
pub unsafe fn render_image(
    grid: *mut GridND<Color, 2>,
    world: &HitKind,
    options: &ImageRenderOptions,
    rand_states: *mut DefaultRand,
) {
    // Safety: 'grid' must point to a valid GridND<Color, 2> that is mutable.
    let grid = unsafe { &mut *grid };

    let (idx_x, idx_y) = thread::index_2d().as_usize_tuple();
    let dims = grid.shape();
    if idx_x >= dims[1] || idx_y >= dims[0] {
        return;
    }
    let px_idx = idx_y * dims[0] + idx_x;

    // generate a tiny offset for the ray for antialiasing
    let rng = unsafe { &mut *rand_states.add(px_idx) };

    let mut pixel_color = Color::black();
    for _ in 0..options.samples_per_pixel {
        // Calculate the pixel sample location.

        use crate::ray::Ray;
        let offset = Vec3::sample_square(rng);
        let pixel_sample = &options.pixel00_loc
            + (&options.pixel_delta_u * (idx_x as Real + offset.x))
            + (&options.pixel_delta_v * (idx_y as Real + offset.y));

        // Apply defocus if enabled
        let ray_origin = &options.origin
            + if options.defocus_angle > 0.0 {
                let offset = Vec3::random_in_unit_disk(rng);
                &options.defocus_disk_u * offset.x + &options.defocus_disk_v * offset.y
            } else {
                Vec3::zero()
            };

        let ray_direction = pixel_sample - &ray_origin;
        let ray = Ray::new(ray_origin, ray_direction);

        pixel_color += ray.color(options.max_depth, world, rng);
    }
    // Store the pixel color in the grid
    *grid.at_mut(idx_y).at_mut(idx_x) = pixel_color / options.samples_per_pixel as Real;
}

#[cfg(target_os = "cuda")]
use crate::hitable::Hitable;
#[cfg(target_os = "cuda")]
use crate::materials::Material;
#[cfg(target_os = "cuda")]
use crate::ray::Ray;

/// This function is used to render the image using CUDA kernels.
/// It functions the same way as `render_image`, but has its inner loop unrolled to improve thread divergence.
/// Improves performance by 40%.
#[cfg(target_os = "cuda")]
#[kernel]
#[cfg(target_os = "cuda")]
pub unsafe fn render_image_v2(
    grid: *mut GridND<Color, 2>,
    world: &HitKind,
    options: &ImageRenderOptions,
    rand_states: *mut DefaultRand,
) {
    // Safety: 'grid' must point to a valid GridND<Color, 2> that is mutable.
    let grid = unsafe { &mut *grid };

    let (idx_x, idx_y) = thread::index_2d().as_usize_tuple();
    let dims = grid.shape();
    if idx_x >= dims[1] || idx_y >= dims[0] {
        return;
    }
    let px_idx = idx_y * dims[0] + idx_x;

    // generate a tiny offset for the ray for antialiasing
    let rng = unsafe { &mut *rand_states.add(px_idx) };

    let mut pixel_color = Color::black();
    let mut cur_sample = 0;
    let mut cur_depth = 0;

    // To satisfy the compiler, we need to initialize `current_ray` here.
    let mut current_ray = Ray::new(Vec3::zero(), Vec3::zero());

    let mut ret = None;
    let mut current_color = Color::white();
    while cur_sample < options.samples_per_pixel {
        if cur_depth == 0 {
            // Calculate the pixel sample location.
            let offset = Vec3::sample_square(rng);
            let pixel_sample = &options.pixel00_loc
                + (&options.pixel_delta_u * (idx_x as Real + offset.x))
                + (&options.pixel_delta_v * (idx_y as Real + offset.y));

            // Apply defocus if enabled
            let ray_origin = &options.origin
                + if options.defocus_angle > 0.0 {
                    let offset = Vec3::random_in_unit_disk(rng);
                    &options.defocus_disk_u * offset.x + &options.defocus_disk_v * offset.y
                } else {
                    Vec3::zero()
                };

            let ray_direction = pixel_sample - &ray_origin;
            current_ray = Ray::new(ray_origin, ray_direction);
            ret = None;
            current_color = Color::white();
        }

        if let Some(hit) = world.hit(&current_ray, &(1e-12..Real::INFINITY)) {
            if let Some((mut scattered_ray, attenuation)) = hit.mat.scatter(&current_ray, hit, rng)
            {
                // Improve the scattered ray's direction and origin.
                // This is to avoid precision issues with re-intersection.
                scattered_ray.direction = scattered_ray.direction.normalize(); // Ensure direction is normalized
                scattered_ray.origin = scattered_ray.origin + &scattered_ray.direction * 1e-4; // Offset to avoid re-intersection

                // Recursively calculate the color of the scattered ray.
                current_ray = scattered_ray;
                current_color = current_color * attenuation;
                cur_depth += 1;
            } else {
                cur_depth = options.max_depth; // Ray was absorbed
            }
        } else {
            let unit_direction = current_ray.direction.normalize();
            let blue = Color::new(0.5, 0.7, 1.0);
            let white = Color::new(1.0, 1.0, 1.0);
            let t = 0.5 * (unit_direction.y + 1.0);
            ret = Some(white.lerp(&blue, t) * &current_color);
            cur_depth = options.max_depth;
        }

        if cur_depth >= options.max_depth {
            // Reset for the next sample
            if let Some(color) = &ret {
                pixel_color += color
            } else {
                // no more light is gathered
            }
            cur_depth = 0;
            cur_sample += 1;
        }
    }
    // Store the pixel color in the grid
    *grid.at_mut(idx_y).at_mut(idx_x) = pixel_color / options.samples_per_pixel as Real;
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
                let pixel_sample = &options.pixel00_loc
                    + (&options.pixel_delta_u * (i as Real + offset.x))
                    + (&options.pixel_delta_v * (j as Real + offset.y));

                // Apply defocus if enabled
                let ray_origin = &options.origin
                    + if options.defocus_angle > 0.0 {
                        let offset = Vec3::random_in_unit_disk();
                        &options.defocus_disk_u * offset.x + &options.defocus_disk_v * offset.y
                    } else {
                        Vec3::zero()
                    };

                let ray_direction = pixel_sample - &ray_origin;
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

#[cfg(target_os = "cuda")]
trait TupleConversion2D {
    fn as_usize_tuple(self) -> (usize, usize);
}

#[cfg(target_os = "cuda")]
impl TupleConversion2D for UVec2 {
    fn as_usize_tuple(self) -> (usize, usize) {
        (self.x as usize, self.y as usize)
    }
}

#[cfg(target_os = "cuda")]
trait TupleConversion3D {
    fn as_usize_tuple(self) -> (usize, usize, usize);
}

#[cfg(target_os = "cuda")]
impl TupleConversion3D for UVec3 {
    fn as_usize_tuple(self) -> (usize, usize, usize) {
        (self.x as usize, self.y as usize, self.z as usize)
    }
}
