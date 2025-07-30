use crate::raytracer::{
    hitable::{HitRecord, Hitable},
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

    pub fn hit(&self, ray: &Ray) -> Option<f32> {
        let co = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = co.dot(ray.direction);
        let c = co.dot(co) - self.radius * self.radius;
        let discriminant = b * b - a * c;

        if discriminant > 0.0 {
            Some((-b - discriminant.sqrt()) / a)
        } else {
            None
        }
    }
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let co = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = co.dot(ray.direction);
        let c = co.dot(co) - self.radius * self.radius;
        let discriminant = b * b - a * c;

        if discriminant > 0.0 {
            let t = (-b - discriminant.sqrt()) / a;
            if t < t_min || t > t_max {
                return None;
            }
            let p = ray.at(t);
            let normal = (p - self.center).normalize();

            Some(HitRecord::new(p, normal, t))
        } else {
            None
        }
    }
}
