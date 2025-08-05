use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{Material, MaterialKind},
    ray::Ray,
    vec3::Vec3,
};

#[cfg(target_os = "cuda")]
use gpu_rand::DefaultRand;

pub struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> MaterialKind {
        MaterialKind::from(Lambertian { albedo })
    }

    fn scatter_inner(
        &self,
        _ray: &Ray,
        hit: &HitRecord,
        random_unit: Vec3,
    ) -> Option<(Ray, Color)> {
        let mut direction = hit.normal + random_unit;
        if direction.near_zero() {
            direction = hit.normal; // Handle near-zero direction to avoid NaN
        }
        let new_ray = Ray::new(hit.p, direction);
        Some((new_ray, self.albedo))
    }
}
impl Material for Lambertian {
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
