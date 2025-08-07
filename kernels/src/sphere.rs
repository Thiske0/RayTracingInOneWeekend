use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitable::{HitRecord, Hitable},
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
    pub fn new_static(center: Point3, radius: Real, material: MaterialKind) -> Self {
        Sphere {
            center: Ray::new(center, Vec3::zero(), 0.0),
            radius,
            mat: material,
        }
    }

    pub fn new_moving(start: Point3, end: Point3, radius: Real, material: MaterialKind) -> Self {
        let velocity = end - &start;
        Sphere {
            center: Ray::new(start, velocity, 0.0),
            radius,
            mat: material,
        }
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

impl IntoBoundingBox for Sphere {
    fn boundingbox(&self) -> BoundingBox {
        let start = self.center.at(0.0);
        let end = self.center.at(1.0);
        let radius_vec = Vec3::new(self.radius, self.radius, self.radius);

        let start_box = BoundingBox::new(&start - &radius_vec, start + &radius_vec);
        let end_box = BoundingBox::new(&end - &radius_vec, end + radius_vec);

        start_box.merge(&end_box)
    }
}
