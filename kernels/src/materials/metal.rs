use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{Material, MaterialKind},
    ray::Ray,
    vec3::{Real, Vec3},
};

#[cfg(target_os = "cuda")]
use gpu_rand::DefaultRand;

pub struct Metal {
    albedo: Color,
    fuzziness: Real,
}

impl Metal {
    pub fn new(albedo: Color, fuzziness: Real) -> MaterialKind {
        MaterialKind::from(Metal { albedo, fuzziness })
    }

    fn scatter_inner(&self, ray: &Ray, hit: &HitRecord, random_unit: Vec3) -> Option<(Ray, Color)> {
        let direction =
            ray.direction.reflect(hit.normal).normalize() + random_unit * self.fuzziness;
        if direction.near_zero() || direction.dot(hit.normal) < 0.0 {
            return None; // Ray is absorbed
        }
        let new_ray = Ray::new(hit.p, direction);
        Some((new_ray, self.albedo))
    }
}
impl Material for Metal {
    #[cfg(target_os = "cuda")]
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        rng: &mut DefaultRand,
    ) -> Option<(Ray, Color)> {
        self.scatter_inner(ray, hit_record, Vec3::random_unit(rng))
    }

    #[cfg(not(target_os = "cuda"))]
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Ray, Color)> {
        self.scatter_inner(ray, hit_record, Vec3::random_unit())
    }
}
