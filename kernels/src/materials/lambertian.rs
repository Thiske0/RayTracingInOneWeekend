use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
    textures::{Texture, TextureKind},
    vec3::Vec3,
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct Lambertian {
    texture: TextureKind,
}

impl Lambertian {
    pub fn new(texture: TextureKind) -> MaterialKind {
        MaterialKind::from(Lambertian { texture })
    }
}
impl Material for Lambertian {
    fn scatter(&self, ray: &Ray, hit: HitRecord, rng: &mut Random) -> Option<(Ray, &Color)> {
        let mut direction = &hit.normal + Vec3::random_unit(rng);
        if direction.near_zero() {
            direction = hit.normal; // Handle near-zero direction to avoid NaN
        }
        let color = self.texture.color(hit.u, hit.v, &hit.p, rng);
        let new_ray = Ray::new(hit.p, direction, ray.time);
        Some((new_ray, color))
    }
}
