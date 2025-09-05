use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
    textures::{Texture, TextureKind, TextureKindDevice},
    vec3::Vec3,
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
    fn scatter(&self, ray: &Ray, hit: &HitRecord, rng: &mut Random) -> Option<(Ray, Color)> {
        let mut direction = &hit.normal + Vec3::random_unit(rng);
        if direction.near_zero() {
            direction = hit.normal.clone(); // Handle near-zero direction to avoid NaN
        }
        let color = self.texture.color(hit.u, hit.v, &hit.p, rng);
        let new_ray = Ray::new(hit.p.clone(), direction, ray.time);
        Some((new_ray, color))
    }

    fn emission(&self, _hit_record: &HitRecord, _rng: &mut Random) -> Color {
        Color::black()
    }
}
