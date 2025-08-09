use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{Material, MaterialKind},
    random::{Random, random_single},
    ray::Ray,
    textures::{Texture, TextureKind, solid::SolidTexture},
    vec3::Real,
};

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct Dielectric {
    refraction_index: Real,
    texture: TextureKind,
}

impl Dielectric {
    pub fn new(refraction_index: Real) -> MaterialKind {
        MaterialKind::from(Dielectric {
            refraction_index,
            texture: SolidTexture::new(Color::white()).into(),
        })
    }

    // Use Schlick's approximation for reflectance.
    fn reflectance(cosine: Real, refraction_index: Real) -> Real {
        let r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}
impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit: HitRecord, rng: &mut Random) -> Option<(Ray, &Color)> {
        let ri = if hit.is_front_face {
            1.0 / self.refraction_index
        } else {
            self.refraction_index
        };

        let unit_direction = ray.direction.normalize();

        let cos_theta = Real::min(-unit_direction.dot(&hit.normal), 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let cannot_refract = ri * sin_theta > 1.0;
        let direction = if cannot_refract
            || Dielectric::reflectance(cos_theta, ri) > random_single(0.0..1.0, rng)
        {
            // Reflect
            unit_direction.reflect(&hit.normal)
        } else {
            // Refract
            unit_direction.refract(&hit.normal, ri)
        };

        let color = self.texture.color(hit.u, hit.v, &hit.p, rng);
        let scattered = Ray::new(hit.p, direction, ray.time);
        Some((scattered, color))
    }
}
