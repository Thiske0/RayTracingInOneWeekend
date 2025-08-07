use crate::{
    ImageRenderOptions,
    color::Color,
    hitable::{HitKind, Hitable},
    materials::Material,
    random::{Random, random_single},
    vec3::{Point3, Real, Vec3},
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: Real,
}

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
    pub fn color(self, depth: usize, hitable: &HitKind, rng: &mut Random) -> Color {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth <= 0 {
            return Color::black();
        }

        let mut current_color = Color::white();
        let mut current_ray = self;

        for _ in 0..depth {
            if let Some(hit) = hitable.hit(&current_ray, &(1e-12..Real::INFINITY)) {
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
                } else {
                    return Color::black(); // Ray was absorbed
                }
            } else {
                let unit_direction = current_ray.direction.normalize();
                let blue = Color::new(0.5, 0.7, 1.0);
                let white = Color::new(1.0, 1.0, 1.0);
                let t = 0.5 * (unit_direction.y + 1.0);
                return white.lerp(&blue, t) * current_color;
            }
        }
        // no more light is gathered
        Color::black()
    }

    pub fn get_ray(i: usize, j: usize, options: &ImageRenderOptions, rng: &mut Random) -> Ray {
        let offset = Vec3::sample_square(rng);
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
        Ray::new(ray_origin, ray_direction, random_single(0.0..1.0, rng))
    }
}
