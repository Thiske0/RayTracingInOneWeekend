use core::ops::Range;

use enum_dispatch::enum_dispatch;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{
        hitable_list::{HitableList, HitableListDevice},
        planar::{Quad, QuadDevice, Triangle, TriangleDevice},
        rotate::{Rotate, RotateDevice},
        sphere::{Sphere, SphereDevice},
        translate::{Translate, TranslateDevice},
    },
    materials::MaterialKind,
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};
use gpu_builder::derive_builder;

pub mod hitable_list;
pub mod planar;
pub mod rotate;
pub mod sphere;
pub mod translate;

#[cfg(not(target_os = "cuda"))]
pub mod hitable_list_builder;

#[enum_dispatch]
pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>>;
}

#[repr(C)]
#[derive_builder('b)]
#[enum_dispatch(Hitable, IntoBoundingBox)]
pub enum HitKind<'b> {
    Sphere(Sphere<'b>),
    Quad(Quad<'b>),
    Triangle(Triangle<'b>),
    HitKindRecursive(HitKindRecursive<'b>),
}

#[repr(C)]
#[derive_builder('b)]
#[enum_dispatch(Hitable, IntoBoundingBox)]
pub enum HitKindRecursive<'b> {
    HitableList(HitableList<'b>),
    Translate(Translate<'b>),
    Rotate(Rotate<'b>),
}

impl<'a> From<HitableList<'a>> for HitKind<'a> {
    fn from(value: HitableList<'a>) -> Self {
        HitKind::HitKindRecursive(HitKindRecursive::HitableList(value))
    }
}

impl<'a> From<Rotate<'a>> for HitKind<'a> {
    fn from(value: Rotate<'a>) -> Self {
        HitKind::HitKindRecursive(HitKindRecursive::Rotate(value))
    }
}

impl<'a> From<Translate<'a>> for HitKind<'a> {
    fn from(value: Translate<'a>) -> Self {
        HitKind::HitKindRecursive(HitKindRecursive::Translate(value))
    }
}

pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub t: Real,
    pub is_front_face: bool,
    pub mat: &'a MaterialKind<'a>,
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
        (u, v): (Real, Real),
    ) -> Self {
        HitRecord {
            p,
            normal,
            t,
            is_front_face,
            mat,
            u,
            v,
        }
    }
}
