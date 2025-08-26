use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{
        dielectric::{Dielectric, DielectricDevice},
        lambertian::{Lambertian, LambertianDevice},
        metal::{Metal, MetalDevice},
    },
    random::Random,
    ray::Ray,
};
use enum_dispatch::enum_dispatch;
use gpu_builder::Builder;

#[enum_dispatch]
pub trait Material {
    /// Returns the scattered ray and the attenuation color.
    fn scatter(&self, ray: &Ray, hit_record: HitRecord, rng: &mut Random) -> Option<(Ray, Color)>;
}

#[repr(C)]
#[derive(Builder)]
#[use_lifetime("'a")]
#[enum_dispatch(Material)]
pub enum MaterialKind<'a> {
    Lambertian(Lambertian<'a>),
    Metal(Metal<'a>),
    Dielectric(Dielectric<'a>),
}

pub mod dielectric;
pub mod lambertian;
pub mod metal;
