use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitRecord, Hitable},
    materials::{MaterialKind, MaterialKindDevice},
    onb::ONB,
    random::{Random, RandomRange},
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};
use gpu_builder::derive_builder;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
pub struct Sphere<'a> {
    center: Ray,
    radius: Real,
    mat: MaterialKind<'a>,
}

impl<'a> Sphere<'a> {
    pub fn new_static(center: Point3, radius: Real, material: MaterialKind<'a>) -> HitKind<'a> {
        Sphere {
            center: Ray::new(center, Vec3::zero(), 0.0),
            radius,
            mat: material,
        }
        .into()
    }

    pub fn new_moving(
        start: Point3,
        end: Point3,
        radius: Real,
        material: MaterialKind<'a>,
    ) -> HitKind<'a> {
        let velocity = end - &start;
        Sphere {
            center: Ray::new(start, velocity, 0.0),
            radius,
            mat: material,
        }
        .into()
    }

    fn get_uv(p: &Point3) -> (Real, Real) {
        let theta = (-p.y).acos();
        let phi = (-p.z).atan2(p.x) + (core::f32::consts::PI as Real);

        let u = phi / (2.0 * (core::f32::consts::PI as Real));
        let v = theta / (core::f32::consts::PI as Real);
        (u, v)
    }

    fn random_to_sphere(&self, distance_squared: Real, rng: &mut Random) -> Vec3 {
        let r1 = rng.random_range(0.0..1.0);
        let r2 = rng.random_range(0.0..1.0);
        let z = 1.0 + r2 * (1.0 - self.radius * self.radius / distance_squared).sqrt();

        let phi = 2.0 * (core::f32::consts::PI as Real) * r1;
        let x = phi.cos() * (1.0 - z * z).sqrt();
        let y = phi.sin() * (1.0 - z * z).sqrt();

        Vec3::new(x, y, z)
    }
}

impl Hitable for Sphere<'_> {
    fn hit<'a>(
        &'a self,
        ray: &Ray,
        range: &Range<Real>,
        _rng: &mut Random,
    ) -> Option<HitRecord<'a>> {
        let actual_center = self.center.at(ray.time);

        let oc = &actual_center - &ray.origin;
        let a = ray.direction.dot(&ray.direction);
        let b = oc.dot(&ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = b * b - a * c;

        if discriminant > 0.0 {
            let mut t = (b - discriminant.sqrt()) / a;
            if !range.contains(&t) {
                t = (b + discriminant.sqrt()) / a;
                if !range.contains(&t) {
                    return None;
                }
            }
            let p = ray.at(t);
            let mut normal = (&p - &actual_center).normalize();
            let uv = Self::get_uv(&normal);

            let is_front_face = normal.dot(&ray.direction) < 0.0;
            if !is_front_face {
                normal = -normal;
            };

            Some(HitRecord::new(p, normal, t, is_front_face, &self.mat, uv))
        } else {
            None
        }
    }

    fn pdf_value(&self, origin: &Point3, direction: &Vec3, rng: &mut Random) -> Real {
        if self.center.direction != Vec3::zero() {
            // This method only works for stationary spheres.
            unimplemented!();
        }
        if let Some(_rec) = self.hit(
            &Ray::new(origin.clone(), direction.clone(), 0.0),
            &Range {
                start: 0.001,
                end: Real::INFINITY,
            },
            rng,
        ) {
            let center = self.center.at(0.0);
            let dist_squared = (center - origin).length_squared();
            let cos_theta_max = (1.0 - self.radius * self.radius / dist_squared).sqrt();
            let solid_angle = 2.0 * (core::f32::consts::PI as Real) * (1.0 - cos_theta_max);

            1.0 / solid_angle
        } else {
            0.0
        }
    }

    fn random(&self, origin: &Point3, rng: &mut Random) -> Vec3 {
        if self.center.direction != Vec3::zero() {
            // This method only works for stationary spheres.
            unimplemented!();
        }
        let center = self.center.at(0.0);
        let direction = center - origin;
        let distance_squared = direction.length_squared();
        let uvw = ONB::new_from_normal(&direction);
        uvw.to_world(&self.random_to_sphere(distance_squared, rng))
    }
}

impl IntoBoundingBox for Sphere<'_> {
    fn boundingbox(&self) -> BoundingBox {
        let start = self.center.at(0.0);
        let end = self.center.at(1.0);
        let radius_vec = Vec3::new(self.radius, self.radius, self.radius);

        let start_box = BoundingBox::new(&start - &radius_vec, start + &radius_vec);
        let end_box = BoundingBox::new(&end - &radius_vec, end + radius_vec);

        start_box.merge(&end_box)
    }
}
