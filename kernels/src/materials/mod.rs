use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{
        dielectric::{Dielectric, DielectricDevice},
        diffuse_light::{DiffuseLight, DiffuseLightDevice},
        is_front::IsFront,
        isotropic::Isotropic,
        lambertian::{Lambertian, LambertianDevice},
        metal::{Metal, MetalDevice},
    },
    pdf::PDFKind,
    random::Random,
    ray::Ray,
};
use enum_dispatch::enum_dispatch;
use gpu_builder::derive_builder;

pub enum ScatterResult<'b, 'a> {
    Scattered(Ray),
    PDF(PDFKind<'b, 'a>),
    None,
}

#[enum_dispatch]
pub trait Material {
    /// Returns the scattered ray and the attenuation color.
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord, rng: &mut Random) -> Option<Color>;
    fn emission(&self, hit_record: &HitRecord, rng: &mut Random) -> Color;
    fn scattering_pdf<'b, 'c>(
        &'b self,
        ray: &Ray,
        hit_record: &HitRecord<'c>,
        rng: &mut Random,
    ) -> ScatterResult<'b, 'c>;
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
    Isotropic(Isotropic),
    IsFront(IsFront),
}

pub mod dielectric;
pub mod diffuse_light;
pub mod is_front;
pub mod isotropic;
pub mod lambertian;
pub mod metal;
