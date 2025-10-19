use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind, ScatterResult},
    random::Random,
    ray::Ray,
    textures::{Texture, TextureKind, TextureKindDevice},
    vec3::{Real, Vec3},
};

use gpu_builder::derive_builder;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
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
    fn scatter(&self, _ray: &Ray, hit: &HitRecord, rng: &mut Random) -> Option<Color> {
        let color = self.texture.color(hit.u, hit.v, &hit.p, rng);
        Some(color)
    }

    fn scattering_pdf<'a, 'b>(
        &'a self,
        ray: &Ray,
        hit: &HitRecord<'b>,
        rng: &mut Random,
    ) -> ScatterResult<'a, 'b> {
        let direction = ray.direction.reflect(&hit.normal).normalize()
            + Vec3::random_unit(rng) * self.fuzziness;
        if direction.near_zero() || direction.dot(&hit.normal) < 0.0 {
            ScatterResult::None
        } else {
            ScatterResult::Scattered(Ray::new(hit.p.clone(), direction, ray.time))
        }
    }

    fn emission(&self, _hit_record: &HitRecord, _rng: &mut Random) -> Color {
        Color::black()
    }
}
