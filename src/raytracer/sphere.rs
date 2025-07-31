use std::ops::Range;

use crate::raytracer::{
    hitable::{HitRecord, Hitable},
    ray::Ray,
    vec3::{Point3, Real},
};

pub struct Sphere {
    pub center: Point3,
    pub radius: Real,
}

impl Sphere {
    pub fn new(center: Point3, radius: Real) -> Self {
        Sphere { center, radius }
    }
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord> {
        let oc = self.center - ray.origin;
        let a = ray.direction.dot(ray.direction);
        let b = oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
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
            let normal = (p - self.center).normalize();

            Some(HitRecord::new(p, normal, t))
        } else {
            None
        }
    }
}
