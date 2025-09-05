use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{
        dielectric::{Dielectric, DielectricDevice},
        diffuse_light::{DiffuseLight, DiffuseLightDevice},
        lambertian::{Lambertian, LambertianDevice},
        metal::{Metal, MetalDevice},
    },
    random::Random,
    ray::Ray,
};
use enum_dispatch::enum_dispatch;
use gpu_builder::derive_builder;

#[enum_dispatch]
pub trait Material {
    /// Returns the scattered ray and the attenuation color.
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord, rng: &mut Random) -> Option<(Ray, Color)>;
    fn emission(&self, hit_record: &HitRecord, rng: &mut Random) -> Color;
}

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
#[enum_dispatch(Material)]
pub enum MaterialKind<'a> {
    Lambertian(Lambertian<'a>),
    Metal(Metal<'a>),
    Dielectric(Dielectric<'a>),
    DiffuseLight(DiffuseLight<'a>),
}

pub mod dielectric;
pub mod diffuse_light;
pub mod lambertian;
pub mod metal;
