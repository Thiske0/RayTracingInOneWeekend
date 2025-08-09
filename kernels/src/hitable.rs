use core::ops::Range;

use enum_dispatch::enum_dispatch;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitable_list::HitableList,
    materials::MaterialKind,
    ray::Ray,
    sphere::Sphere,
    vec3::{Point3, Real, Vec3},
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[enum_dispatch]
pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>>;
}

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
#[enum_dispatch(Hitable)]
pub enum HitKind<'b> {
    Sphere(Sphere),
    HitableList(HitableList<'b>),
}

impl IntoBoundingBox for HitKind<'_> {
    fn boundingbox(&self) -> BoundingBox {
        match self {
            HitKind::Sphere(sphere) => sphere.boundingbox(),
            HitKind::HitableList(list) => list.boundingbox(),
        }
    }
}

pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub t: Real,
    pub is_front_face: bool,
    pub mat: &'a MaterialKind,
    pub u: Real,
    pub v: Real,
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
            u: 0.0,
            v: 0.0,
        }
    }
}
