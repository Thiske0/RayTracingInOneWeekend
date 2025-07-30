use crate::raytracer::{ray::Ray, vec3::Point3};

pub struct Sphere {
    pub center: Point3,
    pub radius: f32,
}

impl Sphere {
    pub fn new(center: Point3, radius: f32) -> Self {
        Sphere { center, radius }
    }

    pub fn hit(&self, ray: &Ray) -> Option<f32> {
        let oc = self.center - ray.origin;
        let a = ray.direction.dot(ray.direction);
        let b = -2.0 * oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant > 0.0 {
            Some((-b - discriminant.sqrt()) / (2.0 * a))
        } else {
            None
        }
    }
}
