use crate::raytracer::{color::Color, hitable::HitRecord, ray::Ray};

pub trait Material {
    /// Returns the scattered ray and the attenuation color.
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Ray, Color)>;
}

pub mod lambertian;
pub mod metal;
