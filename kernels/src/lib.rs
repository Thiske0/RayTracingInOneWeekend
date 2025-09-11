#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(offset_of_enum)]

#[cfg(target_os = "cuda")]
use cuda_std::glam::{UVec2, UVec3};
#[cfg(target_os = "cuda")]
use cuda_std::prelude::*;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
use gpu_builder::DeviceCopyBuilder;
#[cfg(target_os = "cuda")]
use gpu_rand::DefaultRand;

use crate::random::Random;
use crate::vec3::{Point3, Real, Vec3};
use crate::{
    color::Color,
    hitables::{HitKind, HitRecord, Hitable, RecursiveHitable, init_stack},
};
use grid_nd::GridND;

#[cfg(not(target_os = "cuda"))]
use indicatif::ProgressBar;

pub mod boundingbox;
pub mod color;
pub mod hitables;
pub mod materials;
pub mod random;
pub mod ray;
pub mod textures;
pub mod vec3;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
#[repr(C)]
#[derive(DeviceCopyBuilder)]
pub struct ImageRenderOptions {
    pub samples_per_pixel: usize,
    pub origin: Point3,
    pub max_depth: usize,
    pub background: Color,

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
pub unsafe fn render_image<'a>(
    grid: *mut GridND<Color, 2>,
    world: &'a HitKind<'a>,
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
pub fn render_image<'a, 'b: 'a>(
    grid: &mut GridND<Color, 2>,
    world: &'b HitKind<'a>,
    options: &ImageRenderOptions,
) {
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

fn render_pixel<'a>(
    i: usize,
    j: usize,
    options: &ImageRenderOptions,
    world: &HitKind<'a>,
    rng: &mut Random,
) -> Color {
    let mut pixel_color = Color::black();
    for _ in 0..options.samples_per_pixel {
        let ray = Ray::get_ray(i, j, options, rng);
        pixel_color += ray.color(world, options, rng);
    }
    pixel_color / options.samples_per_pixel as Real
}
use crate::materials::Material;

/// This function is used to render the image using CUDA kernels.
/// It functions the same way as `render_image`, but has its inner loop unrolled to improve thread divergence.
/// Improves performance by 40%.
#[allow(unused)]
fn render_pixel_v2<'a>(
    i: usize,
    j: usize,
    options: &ImageRenderOptions,
    world: &HitKind<'a>,
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

        if let Some(hit) = world.hit(current_ray.clone(), 1e-12..Real::INFINITY, rng) {
            pixel_color += (&current_color) * hit.mat.emission(&hit, rng);
            if let Some((mut scattered_ray, attenuation)) = hit.mat.scatter(&current_ray, &hit, rng)
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
            // update color
            pixel_color += (&options.background) * &current_color;
            cur_depth = 0;
            cur_sample += 1;
        }
    }
    pixel_color / options.samples_per_pixel as Real
}

/// This function is used to render the image using CUDA kernels.
/// It functions the same way as `render_image_v2`, but has its inner loop unrolled to improve thread divergence.
/// turns out to be slower than v2
#[allow(unused)]
fn render_pixel_v3<'a>(
    i: usize,
    j: usize,
    options: &ImageRenderOptions,
    world: &'a HitKind<'a>,
    rng: &mut Random,
) -> Color {
    use crate::hitables::STACK_SIZE;
    use crate::hitables::StackEntry;

    let mut pixel_color = Color::black();
    let mut cur_sample = 0;
    let mut cur_depth = 0;

    let mut stack_ptr = 0;
    let mut stack: [StackEntry<'a, '_>; STACK_SIZE] = init_stack::<STACK_SIZE>(world);
    let mut hit_record: Option<HitRecord<'a>> = None;
    let mut range = 1e-12..Real::INFINITY;

    // To satisfy the compiler, we need to initialize `current_ray` here.
    let mut current_ray = Ray::new(Vec3::zero(), Vec3::zero(), 0.0);

    let mut current_color = Color::white();
    while cur_sample < options.samples_per_pixel {
        if stack_ptr == 0 {
            if cur_depth == 0 {
                current_ray = Ray::get_ray(i, j, options, rng);
                current_color = Color::white();
            }

            range = 1e-12..Real::INFINITY;
            match world {
                HitKind::HitKindNonRecursive(h) => {
                    hit_record = h.hit(&current_ray, &range, rng);
                }
                HitKind::HitKindRecursive(h) => {
                    stack[0].next_count = 0; // reset next_count
                    stack_ptr = 1;
                    hit_record = None;
                }
            }
        }

        if stack_ptr > 0 {
            // recurse till stack is empty
            if stack_ptr >= STACK_SIZE {
                // Stack overflow, break to avoid infinite loop
                panic!("Stack overflow in hit_recursive");
            }
            let current = &mut stack[stack_ptr - 1];
            if let Some((inner_hitkind, next_count)) = current.hitkind.hit_recursive(
                &mut current_ray,
                &mut range,
                &mut hit_record,
                current.next_count,
                rng,
            ) {
                current.next_count = next_count;
                match inner_hitkind {
                    HitKind::HitKindNonRecursive(inner_hitkind) => {
                        if let Some(rec) = inner_hitkind.hit(&current_ray, &range, rng) {
                            if let Some(current_rec) = &hit_record {
                                if rec.t < current_rec.t {
                                    range = range.start..rec.t;
                                    hit_record = Some(rec);
                                }
                            } else {
                                range = range.start..rec.t;
                                hit_record = Some(rec);
                            }
                        }
                    }
                    HitKind::HitKindRecursive(inner_hitkind) => {
                        stack_ptr += 1;
                        stack[stack_ptr - 1] = StackEntry::new(&inner_hitkind);
                    }
                }
            } else {
                stack_ptr -= 1;
            }
        }

        if stack_ptr == 0 {
            // process hit
            if let Some(hit) = &hit_record {
                pixel_color += (&current_color) * hit.mat.emission(&hit, rng);
                if let Some((mut scattered_ray, attenuation)) =
                    hit.mat.scatter(&current_ray, hit, rng)
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
                // update color
                pixel_color += (&options.background) * &current_color;
                cur_depth = 0;
                cur_sample += 1;
            }
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
