use std::ops::Range;

use crate::raytracer::{
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};

pub trait Hitable {
    fn hit(&self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord>;
}

pub struct HitRecord {
    pub p: Point3,
    pub normal: Vec3,
    pub t: Real,
}

impl HitRecord {
    pub fn new(p: Point3, normal: Vec3, t: Real) -> Self {
        HitRecord { p, normal, t }
    }
}
