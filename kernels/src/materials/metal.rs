use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
    vec3::{Real, Vec3},
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct Metal {
    albedo: Color,
    fuzziness: Real,
}

impl Metal {
    pub fn new(albedo: Color, fuzziness: Real) -> MaterialKind {
        MaterialKind::from(Metal { albedo, fuzziness })
    }
}
impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit_record: HitRecord, rng: &mut Random) -> Option<(Ray, &Color)> {
        let direction = ray.direction.reflect(&hit_record.normal).normalize()
            + Vec3::random_unit(rng) * self.fuzziness;
        if direction.near_zero() || direction.dot(&hit_record.normal) < 0.0 {
            return None; // Ray is absorbed
        }
        let new_ray = Ray::new(hit_record.p, direction, ray.time);
        Some((new_ray, &self.albedo))
    }
}
