use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{dielectric::Dielectric, lambertian::Lambertian, metal::Metal},
    random::Random,
    ray::Ray,
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait Material {
    /// Returns the scattered ray and the attenuation color.
    fn scatter(&self, ray: &Ray, hit_record: HitRecord, rng: &mut Random) -> Option<(Ray, &Color)>;
}

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
#[enum_dispatch(Material)]
pub enum MaterialKind {
    Lambertian(Lambertian),
    Metal(Metal),
    Dielectric(Dielectric),
}

pub mod dielectric;
pub mod lambertian;
pub mod metal;
