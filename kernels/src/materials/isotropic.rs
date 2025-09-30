use crate::{
    color::Color,
    hitables::HitRecord,
    materials::{Material, MaterialKind},
    random::Random,
    ray::Ray,
    vec3::Vec3,
};
use gpu_builder::DeviceCopyBuilder;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Copy, Clone, DeviceCopy))]
#[derive(DeviceCopyBuilder)]
pub struct Isotropic {
    color: Color,
}

impl Isotropic {
    pub fn new(color: Color) -> MaterialKind<'static> {
        Isotropic { color }.into()
    }
}
impl Material for Isotropic {
    fn scatter(&self, ray: &Ray, hit: &HitRecord, rng: &mut Random) -> Option<(Ray, Color)> {
        let direction = Vec3::random_unit(rng);
        let new_ray = Ray::new(hit.p.clone(), direction, ray.time);
        Some((new_ray, self.color.clone()))
    }

    fn emission(&self, _hit_record: &HitRecord, _rng: &mut Random) -> Color {
        Color::black()
    }
}
