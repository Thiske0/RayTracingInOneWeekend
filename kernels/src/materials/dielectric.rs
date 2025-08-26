use crate::random::RandomRange;
use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
    textures::{Texture, TextureKind, TextureKindDevice, solid::SolidTexture},
    vec3::Real,
};
use gpu_builder::Builder;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[repr(C)]
#[derive(Builder)]
#[use_lifetime("'a")]
pub struct Dielectric<'a> {
    refraction_index: Real,
    texture: TextureKind<'a>,
}

impl<'a> Dielectric<'a> {
    pub fn new(refraction_index: Real) -> MaterialKind<'a> {
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
impl Material for Dielectric<'_> {
    fn scatter(&self, ray: &Ray, hit: HitRecord, rng: &mut Random) -> Option<(Ray, Color)> {
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
            || Dielectric::reflectance(cos_theta, ri) > rng.random_range(0.0..1.0)
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
