use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind, ScatterResult},
    pdf::SpherePDF,
    random::Random,
    ray::Ray,
};
use gpu_builder::DeviceCopyBuilder;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Copy, Clone, DeviceCopy))]
#[derive(DeviceCopyBuilder)]
pub struct Isotropic {
    color: Color,
}

impl Isotropic {
    pub fn new(color: Color) -> MaterialKind<'static> {
        Isotropic { color }.into()
    }
}
impl Material for Isotropic {
    fn scatter(&self, _ray: &Ray, _hit: &HitRecord, _rng: &mut Random) -> Option<Color> {
        Some(self.color.clone())
    }

    fn scattering_pdf<'a, 'b>(
        &'a self,
        _ray: &Ray,
        _hit: &HitRecord<'b>,
        _rng: &mut Random,
    ) -> ScatterResult<'a, 'b> {
        ScatterResult::PDF(SpherePDF::new().into())
    }

    fn emission(&self, _hit_record: &HitRecord, _rng: &mut Random) -> Color {
        Color::black()
    }
}

#[cfg(not(target_os = "cuda"))]
use crate::materials::IsLight;

#[cfg(not(target_os = "cuda"))]
impl IsLight for Isotropic {
    fn is_light(&self) -> bool {
        false
    }
}
