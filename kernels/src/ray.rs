use crate::{
    ImageRenderOptions,
    color::Color,
    hitables::HitKind,
    materials::Material,
    random::Random,
    vec3::{Point3, Real, Vec3},
};
use gpu_builder::DeviceCopyBuilder;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[cfg_attr(not(target_os = "cuda"), derive(Copy, DeviceCopy))]
#[repr(C)]
#[derive(DeviceCopyBuilder, Clone)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: Real,
}

use crate::random::RandomRange;

impl Ray {
    pub fn new(origin: Point3, direction: Vec3, time: Real) -> Self {
        Ray {
            origin,
            direction,
            time,
        }
    }

    pub fn origin(&self) -> &Point3 {
        &self.origin
    }

    pub fn direction(&self) -> &Vec3 {
        &self.direction
    }

    pub fn at(&self, t: Real) -> Point3 {
        &self.origin + &self.direction * t
    }
    pub fn color<'a>(
        self,
        hitable: &HitKind<'a>,
        options: &ImageRenderOptions,
        rng: &mut Random,
    ) -> Color {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if options.max_depth <= 0 {
            return Color::black();
        }

        let mut current_color = Color::white();
        let mut final_color = Color::black();
        let mut current_ray = self;

        for _ in 0..options.max_depth {
            if let Some(hit) = hitable.hit(current_ray.clone(), 1e-12..Real::INFINITY, rng) {
                final_color += (&current_color) * hit.mat.emission(&hit, rng);
                if let Some((mut scattered_ray, attenuation)) =
                    hit.mat.scatter(&current_ray, &hit, rng)
                {
                    // Improve the scattered ray's direction and origin.
                    // This is to avoid precision issues with re-intersection.
                    scattered_ray.direction = scattered_ray.direction.normalize(); // Ensure direction is normalized
                    scattered_ray.origin = scattered_ray.origin + &scattered_ray.direction * 1e-4; // Offset to avoid re-intersection

                    // Recursively calculate the color of the scattered ray.
                    current_ray = scattered_ray;
                    current_color = current_color * attenuation;
                } else {
                    return final_color;
                }
            } else {
                return (&options.background) * current_color + final_color;
            }
        }
        // no more light is gathered
        final_color
    }

    pub fn get_ray(
        cur_sample: usize,
        i: usize,
        j: usize,
        options: &ImageRenderOptions,
        rng: &mut Random,
    ) -> Ray {
        let offset = Self::stratisfied_sample_square(cur_sample, options.samples_per_pixel, rng);
        let pixel_sample = &options.pixel00_loc
            + (&options.pixel_delta_u * (i as Real + offset.x))
            + (&options.pixel_delta_v * (j as Real + offset.y));

        // Apply defocus if enabled
        let ray_origin = &options.origin
            + if options.defocus_angle > 0.0 {
                let offset = Vec3::random_in_unit_disk(rng);
                &options.defocus_disk_u * offset.x + &options.defocus_disk_v * offset.y
            } else {
                Vec3::zero()
            };

        let ray_direction = pixel_sample - &ray_origin;
        Ray::new(ray_origin, ray_direction, rng.random_range(0.0..1.0))
    }

    fn stratisfied_sample_square(cur_sample: usize, amount: usize, rng: &mut Random) -> Vec3 {
        let n = (amount as Real).sqrt().floor() as usize;
        let i = cur_sample % n;
        let j = cur_sample / n;
        let offset = if cur_sample < n * n {
            Vec3::new(
                (i as Real + rng.random_range(0.0..1.0)) / n as Real,
                (j as Real + rng.random_range(0.0..1.0)) / n as Real,
                0.0,
            )
        } else {
            Vec3::sample_square(rng)
        };
        offset
    }
}
