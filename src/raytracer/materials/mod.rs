use crate::raytracer::{
    hitable::HitRecord,
    materials::{dielectric::Dielectric, lambertian::Lambertian, metal::Metal},
    ray::Ray,
};
use simple_ray_tracer_kernels::color::Color;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait Material {
    /// Returns the scattered ray and the attenuation color.
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Ray, Color)>;
}

#[enum_dispatch(Material)]
pub enum MaterialKind {
    Lambertian(Lambertian),
    Metal(Metal),
    Dielectric(Dielectric),
}

pub mod dielectric;
pub mod lambertian;
pub mod metal;
