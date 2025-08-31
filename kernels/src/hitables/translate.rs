use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, Hitable},
    ray::Ray,
    vec3::{Real, Vec3},
};
use gpu_builder::derive_builder;
use ref_builder::{RefBuilder, RefBuilderDevice};

#[repr(C)]
#[derive_builder('a)]
pub struct Translate<'a> {
    offset: Vec3,
    inner: RefBuilder<'a, HitKind<'a>>,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> Translate<'a> {
    pub fn new(offset: Vec3, inner: &'a HitKind<'a>) -> HitKind<'a> {
        Translate {
            offset,
            inner: RefBuilder::new(inner),
        }
        .into()
    }

    pub fn new_owned(offset: Vec3, inner: HitKind<'a>) -> HitKind<'a> {
        Translate {
            offset,
            inner: RefBuilder::new_owned(inner),
        }
        .into()
    }
}

impl Hitable for Translate<'_> {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>> {
        let moved_ray = Ray::new(&ray.origin - &self.offset, ray.direction.clone(), ray.time);
        if let Some(mut rec) = self.inner.hit(&moved_ray, range) {
            rec.p += &self.offset;
            return Some(rec);
        }
        None
    }
}

impl IntoBoundingBox for Translate<'_> {
    fn boundingbox(&self) -> BoundingBox {
        let inner_box = self.inner.boundingbox();
        inner_box.translate(&self.offset)
    }
}
