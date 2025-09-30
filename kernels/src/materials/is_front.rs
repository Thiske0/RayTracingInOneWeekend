use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
};
use gpu_builder::DeviceCopyBuilder;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Copy, Clone, DeviceCopy))]
#[derive(DeviceCopyBuilder)]
pub struct IsFront {}

impl IsFront {
    pub fn new() -> MaterialKind<'static> {
        MaterialKind::from(IsFront {})
    }
}
impl Material for IsFront {
    fn scatter(&self, _ray: &Ray, _hit: &HitRecord, _rng: &mut Random) -> Option<(Ray, Color)> {
        None
    }

    fn emission(&self, hit: &HitRecord, _rng: &mut Random) -> Color {
        if hit.is_front_face {
            Color::white()
        } else {
            Color::black()
        }
    }
}
