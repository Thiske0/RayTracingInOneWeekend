use std::ops::Range;

use enum_dispatch::enum_dispatch;

use crate::raytracer::{
    hitable_list::HitableList,
    materials::MaterialKind,
    ray::Ray,
    sphere::Sphere,
    vec3::{Point3, Real, Vec3},
};

#[enum_dispatch]
pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>>;
}

#[enum_dispatch(Hitable)]
pub enum HitKind {
    Sphere(Sphere),
    HitableList(HitableList),
}

pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub t: Real,
    pub is_front_face: bool,
    pub mat: &'a MaterialKind,
}

impl<'a> HitRecord<'a> {
    pub fn new(
        p: Point3,
        normal: Vec3,
        t: Real,
        is_front_face: bool,
        mat: &'a MaterialKind,
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
