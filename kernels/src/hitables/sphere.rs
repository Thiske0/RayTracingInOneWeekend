use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitRecord, Hitable},
    materials::{MaterialKind, MaterialKindDevice},
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};
use gpu_builder::Builder;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[repr(C)]
#[derive(Builder)]
#[use_lifetime("'a")]
pub struct Sphere<'a> {
    center: Ray,
    radius: Real,
    mat: MaterialKind<'a>,
}

impl<'a> Sphere<'a> {
    pub fn new_static(center: Point3, radius: Real, material: MaterialKind<'a>) -> Self {
        Sphere {
            center: Ray::new(center, Vec3::zero(), 0.0),
            radius,
            mat: material,
        }
    }

    pub fn new_moving(
        start: Point3,
        end: Point3,
        radius: Real,
        material: MaterialKind<'a>,
    ) -> Self {
        let velocity = end - &start;
        Sphere {
            center: Ray::new(start, velocity, 0.0),
            radius,
            mat: material,
        }
    }

    fn get_uv(p: &Point3) -> (Real, Real) {
        let theta = (-p.y).acos();
        let phi = (-p.z).atan2(p.x) + (core::f32::consts::PI as Real);

        let u = phi / (2.0 * (core::f32::consts::PI as Real));
        let v = theta / (core::f32::consts::PI as Real);
        (u, v)
    }
}

impl Hitable for Sphere<'_> {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>> {
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
