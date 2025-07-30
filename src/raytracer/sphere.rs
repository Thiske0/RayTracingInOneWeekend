use crate::raytracer::{
    hitable::{HitRecord, Hitable},
    interval::Interval,
    ray::Ray,
    vec3::Point3,
};

pub struct Sphere {
    pub center: Point3,
    pub radius: f32,
}

impl Sphere {
    pub fn new(center: Point3, radius: f32) -> Self {
        Sphere { center, radius }
    }
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, interval: &Interval) -> Option<HitRecord> {
        let oc = self.center - ray.origin;
        let a = ray.direction.dot(ray.direction);
        let b = oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - a * c;

        if discriminant > 0.0 {
            let mut t = (b - discriminant.sqrt()) / a;
            if !interval.contains(t) {
                t = (b + discriminant.sqrt()) / a;
                if !interval.contains(t) {
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
