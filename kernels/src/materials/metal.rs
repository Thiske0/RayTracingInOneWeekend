use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
    textures::{Texture, TextureKind, TextureKindDevice},
    vec3::{Real, Vec3},
};

use gpu_builder::Builder;

#[repr(C)]
#[derive(Builder)]
#[use_lifetime("'a")]
pub struct Metal<'a> {
    texture: TextureKind<'a>,
    fuzziness: Real,
}

impl Metal<'_> {
    pub fn new(texture: TextureKind, fuzziness: Real) -> MaterialKind {
        MaterialKind::from(Metal { texture, fuzziness })
    }
}
impl Material for Metal<'_> {
    fn scatter(&self, ray: &Ray, hit: HitRecord, rng: &mut Random) -> Option<(Ray, &Color)> {
        let direction = ray.direction.reflect(&hit.normal).normalize()
            + Vec3::random_unit(rng) * self.fuzziness;
        if direction.near_zero() || direction.dot(&hit.normal) < 0.0 {
            return None; // Ray is absorbed
        }
        let color = self.texture.color(hit.u, hit.v, &hit.p, rng);
        let new_ray = Ray::new(hit.p, direction, ray.time);
        Some((new_ray, color))
    }
}
