use std::ops::Range;

use crate::raytracer::{
    hitable::{HitRecord, Hitable},
    materials::Material,
    ray::Ray,
    vec3::{Point3, Real},
};

pub struct Sphere<T: Material> {
    center: Point3,
    radius: Real,
    mat: T,
}

impl<T: Material> Sphere<T> {
    pub fn new(center: Point3, radius: Real, material: T) -> Self {
        Sphere {
            center,
            radius,
            mat: material,
        }
    }
}

impl<T: Material> Hitable for Sphere<T> {
    fn hit(&self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord> {
        let oc = self.center - ray.origin;
        let a = ray.direction.dot(ray.direction);
        let b = oc.dot(ray.direction);
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
            let mut normal = (p - self.center).normalize();

            let is_front_face = normal.dot(ray.direction) < 0.0;
            if !is_front_face {
                normal = -normal;
            };

            Some(HitRecord::new(p, normal, t, is_front_face, &self.mat))
        } else {
            None
        }
    }
}
