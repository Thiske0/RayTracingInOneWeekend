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

use crate::random::Random;
use crate::vec3::{Point3, Real, Vec3};
use crate::{color::Color, hitable::HitKind};
use grid_nd::GridND;

#[cfg(not(target_os = "cuda"))]
use indicatif::ProgressBar;

pub mod boundingbox;
pub mod color;
pub mod hitable;
pub mod hitable_list;
pub mod materials;
pub mod random;
pub mod ray;
pub mod sphere;
pub mod textures;
pub mod vec3;

#[cfg(not(target_os = "cuda"))]
pub mod hitable_list_builder;

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
    let mut rng = unsafe { &mut *rand_states.add(px_idx) };

    // Store the pixel color in the grid
    *grid.at_mut(idx_y).at_mut(idx_x) = render_pixel_v2(idx_x, idx_y, options, world, &mut rng);
}
use crate::ray::Ray;

#[cfg(not(target_os = "cuda"))]
pub fn render_image(grid: &mut GridND<Color, 2>, world: &HitKind, options: &ImageRenderOptions) {
    // Set up the progress bar

    let progress = ProgressBar::new(grid.shape().iter().product::<usize>() as u64);

    let mut rng = rand::rng();
    for i in 0..grid.shape()[1] {
        for j in 0..grid.shape()[0] {
            *grid.at_mut(j).at_mut(i) = render_pixel(i, j, options, world, &mut rng);
        }
        progress.inc(grid.shape()[0] as u64);
    }
    progress.finish();
}

fn render_pixel(
    i: usize,
    j: usize,
    options: &ImageRenderOptions,
    world: &HitKind,
    rng: &mut Random,
) -> Color {
    let mut pixel_color = Color::black();
    for _ in 0..options.samples_per_pixel {
        let ray = Ray::get_ray(i, j, options, rng);
        pixel_color += ray.color(options.max_depth, world, rng);
    }
    pixel_color / options.samples_per_pixel as Real
}
use crate::hitable::Hitable;
use crate::materials::Material;

/// This function is used to render the image using CUDA kernels.
/// It functions the same way as `render_image`, but has its inner loop unrolled to improve thread divergence.
/// Improves performance by 40%.
#[allow(unused)]
fn render_pixel_v2(
    i: usize,
    j: usize,
    options: &ImageRenderOptions,
    world: &HitKind,
    rng: &mut Random,
) -> Color {
    let mut pixel_color = Color::black();
    let mut cur_sample = 0;
    let mut cur_depth = 0;

    // To satisfy the compiler, we need to initialize `current_ray` here.
    let mut current_ray = Ray::new(Vec3::zero(), Vec3::zero(), 0.0);

    let mut current_color = Color::white();
    while cur_sample < options.samples_per_pixel {
        if cur_depth == 0 {
            current_ray = Ray::get_ray(i, j, options, rng);
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

                if cur_depth >= options.max_depth {
                    // max depth is reached, don't gather any light
                    cur_depth = 0;
                    cur_sample += 1;
                }
            } else {
                // Ray was absorbed
                cur_depth = 0;
                cur_sample += 1;
            }
        } else {
            let unit_direction = current_ray.direction.normalize();
            let blue = Color::new(0.5, 0.7, 1.0);
            let white = Color::new(1.0, 1.0, 1.0);
            let t = 0.5 * (unit_direction.y + 1.0);

            // update color
            pixel_color += white.lerp(&blue, t) * &current_color;
            cur_depth = 0;
            cur_sample += 1;
        }
    }
    pixel_color / options.samples_per_pixel as Real
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
