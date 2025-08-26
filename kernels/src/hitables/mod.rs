use core::ops::Range;

use enum_dispatch::enum_dispatch;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::hitable_list::{HitableList, HitableListDevice},
    hitables::sphere::{Sphere, SphereDevice},
    materials::MaterialKind,
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};
use gpu_builder::Builder;

pub mod hitable_list;
pub mod planar;
pub mod sphere;

#[cfg(not(target_os = "cuda"))]
pub mod hitable_list_builder;

#[enum_dispatch]
pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>>;
}

#[repr(C)]
#[derive(Builder)]
#[use_lifetime("'b")]
#[enum_dispatch(Hitable)]
pub enum HitKind<'b> {
    Sphere(Sphere<'b>),
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
