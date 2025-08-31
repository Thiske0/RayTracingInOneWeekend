use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind},
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
    fn scatter(&self, _ray: &Ray, _hit: HitRecord, _rng: &mut Random) -> Option<(Ray, Color)> {
        None
    }

    fn emission(&self, hit: &HitRecord, rng: &mut Random) -> Color {
        self.texture.color(hit.u, hit.v, &hit.p, rng)
    }
}
