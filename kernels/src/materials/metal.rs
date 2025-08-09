use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
    textures::{Texture, TextureKind},
    vec3::{Real, Vec3},
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct Metal {
    texture: TextureKind,
    fuzziness: Real,
}

impl Metal {
    pub fn new(texture: TextureKind, fuzziness: Real) -> MaterialKind {
        MaterialKind::from(Metal { texture, fuzziness })
    }
}
impl Material for Metal {
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
