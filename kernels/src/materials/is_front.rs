use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind, ScatterResult},
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
    fn scatter(&self, _ray: &Ray, _hit: &HitRecord, _rng: &mut Random) -> Option<Color> {
        None
    }

    fn scattering_pdf<'a, 'b>(
        &'a self,
        _ray: &Ray,
        _hit_record: &HitRecord<'b>,
        _rng: &mut Random,
    ) -> ScatterResult<'a, 'b> {
        unreachable!()
    }

    fn emission(&self, hit: &HitRecord, _rng: &mut Random) -> Color {
        if hit.is_front_face {
            Color::white()
        } else {
            Color::black()
        }
    }
}

#[cfg(not(target_os = "cuda"))]
use crate::materials::IsLight;

#[cfg(not(target_os = "cuda"))]
impl IsLight for IsFront {
    fn is_light(&self) -> bool {
        false
    }
}
