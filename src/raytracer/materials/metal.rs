use crate::raytracer::{color::Color, hitable::HitRecord, materials::Material, ray::Ray};

pub struct Metal {
    albedo: Color,
}

impl Metal {
    pub fn new(albedo: Color) -> Self {
        Metal { albedo }
    }
}
impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<(Ray, Color)> {
        let direction = ray.direction.reflect(hit.normal);
        let new_ray = Ray::new(hit.p, direction);
        Some((new_ray, self.albedo))
    }
}
