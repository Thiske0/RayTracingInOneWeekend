use crate::raytracer::{
    ray::Ray,
    vec3::{Point3, Vec3},
};

pub trait Hitable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

pub struct HitRecord {
    pub p: Point3,
    pub normal: Vec3,
    pub t: f32,
}

impl HitRecord {
    pub fn new(p: Point3, normal: Vec3, t: f32) -> Self {
        HitRecord { p, normal, t }
    }
}
