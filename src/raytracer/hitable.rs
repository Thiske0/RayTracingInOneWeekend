use std::ops::Range;

use crate::raytracer::{
    materials::Material,
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};

pub trait Hitable {
    fn hit(&self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord>;
}

pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub t: Real,
    pub mat: &'a dyn Material,
}

impl<'a> HitRecord<'a> {
    pub fn new(p: Point3, normal: Vec3, t: Real, mat: &'a dyn Material) -> Self {
        HitRecord { p, normal, t, mat }
    }
}
