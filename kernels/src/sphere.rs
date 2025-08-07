use core::ops::Range;

use crate::{
    hitable::{HitKind, HitRecord, Hitable},
    materials::MaterialKind,
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct Sphere {
    center: Ray,
    radius: Real,
    mat: MaterialKind,
}

impl Sphere {
    pub fn new_static<'a>(center: Point3, radius: Real, material: MaterialKind) -> HitKind<'a> {
        HitKind::from(Sphere {
            center: Ray::new(center, Vec3::zero(), 0.0),
            radius,
            mat: material,
        })
    }

    pub fn new_moving<'a>(
        start: Point3,
        end: Point3,
        radius: Real,
        material: MaterialKind,
    ) -> HitKind<'a> {
        let velocity = end - &start;
        HitKind::from(Sphere {
            center: Ray::new(start, velocity, 0.0),
            radius,
            mat: material,
        })
    }
}

impl Hitable for Sphere {
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

            let is_front_face = normal.dot(&ray.direction) < 0.0;
            if !is_front_face {
                normal = -normal;
            };

            Some(HitRecord::new(p, normal, t, is_front_face, &self.mat))
        } else {
            None
        }
    }
}
