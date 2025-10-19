use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind, ScatterResult},
    random::Random,
    ray::Ray,
    textures::{Texture, TextureKind, TextureKindDevice},
};

use gpu_builder::derive_builder;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
pub struct DiffuseLight<'a> {
    texture: TextureKind<'a>,
}

impl DiffuseLight<'_> {
    pub fn new(texture: TextureKind) -> MaterialKind {
        MaterialKind::from(DiffuseLight { texture })
    }
}
impl Material for DiffuseLight<'_> {
    fn scatter(&self, _ray: &Ray, _hit: &HitRecord, _rng: &mut Random) -> Option<Color> {
        None
    }

    fn scattering_pdf<'a, 'b>(
        &'a self,
        _ray: &Ray,
        _hit: &HitRecord<'b>,
        _rng: &mut Random,
    ) -> ScatterResult<'a, 'b> {
        unreachable!()
    }

    fn emission(&self, hit: &HitRecord, rng: &mut Random) -> Color {
        if hit.is_front_face {
            self.texture.color(hit.u, hit.v, &hit.p, rng)
        } else {
            Color::black()
        }
    }
}

#[cfg(not(target_os = "cuda"))]
use crate::materials::IsLight;

#[cfg(not(target_os = "cuda"))]
impl IsLight for DiffuseLight<'_> {
    fn is_light(&self) -> bool {
        true
    }
}
