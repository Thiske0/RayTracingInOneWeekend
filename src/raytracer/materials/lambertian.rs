use crate::raytracer::{
    color::Color, hitable::HitRecord, materials::Material, ray::Ray, vec3::Vec3,
};

pub struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Self {
        Lambertian { albedo }
    }
}
impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, hit: &HitRecord) -> Option<(Ray, Color)> {
        let mut direction = hit.normal + Vec3::random_unit();
        if direction.near_zero() {
            direction = hit.normal; // Handle near-zero direction to avoid NaN
        }
        let new_ray = Ray::new(hit.p, direction);
        Some((new_ray, self.albedo))
    }
}
