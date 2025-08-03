use std::ops::Range;

use crate::raytracer::{
    materials::Material,
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};

pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>>;
}

pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub t: Real,
    pub is_front_face: bool,
    pub mat: &'a dyn Material,
}

impl<'a> HitRecord<'a> {
    pub fn new(
        p: Point3,
        normal: Vec3,
        t: Real,
        is_front_face: bool,
        mat: &'a dyn Material,
    ) -> Self {
        HitRecord {
            p,
            normal,
            t,
            is_front_face,
            mat,
        }
    }
}
