use crate::{
    color::Color,
    hitable::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
    vec3::Vec3,
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> MaterialKind {
        MaterialKind::from(Lambertian { albedo })
    }
}
impl Material for Lambertian {
    fn scatter(
        &self,
        _ray: &Ray,
        hit_record: HitRecord,
        rng: &mut Random,
    ) -> Option<(Ray, &Color)> {
        let mut direction = &hit_record.normal + Vec3::random_unit(rng);
        if direction.near_zero() {
            direction = hit_record.normal; // Handle near-zero direction to avoid NaN
        }
        let new_ray = Ray::new(hit_record.p, direction);
        Some((new_ray, &self.albedo))
    }
}
