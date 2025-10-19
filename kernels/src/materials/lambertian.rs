use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind, ScatterResult},
    pdf::CosinePDF,
    random::Random,
    ray::Ray,
    textures::{Texture, TextureKind, TextureKindDevice},
};

use gpu_builder::derive_builder;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
pub struct Lambertian<'a> {
    texture: TextureKind<'a>,
}

impl Lambertian<'_> {
    pub fn new(texture: TextureKind) -> MaterialKind {
        MaterialKind::from(Lambertian { texture })
    }
}
impl Material for Lambertian<'_> {
    fn scatter(&self, _ray: &Ray, hit: &HitRecord, rng: &mut Random) -> Option<Color> {
        let color = self.texture.color(hit.u, hit.v, &hit.p, rng);
        Some(color)
    }

    fn scattering_pdf<'a, 'b>(
        &'a self,
        _ray: &Ray,
        hit: &HitRecord<'b>,
        _rng: &mut Random,
    ) -> ScatterResult<'a, 'b> {
        ScatterResult::PDF(CosinePDF::new(&hit.normal))
    }

    fn emission(&self, _hit_record: &HitRecord, _rng: &mut Random) -> Color {
        Color::black()
    }
}
