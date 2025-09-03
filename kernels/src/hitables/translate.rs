use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, RecursiveHitable},
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

impl RecursiveHitable for Translate<'_> {
    fn hit_recursive<'a>(
        &'a self,
        ray: &mut Ray,
        _range: &Range<Real>,
        hit_record: &mut Option<HitRecord<'a>>,
        count: usize,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count == 0 {
            if let Some(rec) = hit_record {
                rec.p -= &self.offset;
            }
            *ray = Ray::new(&ray.origin - &self.offset, ray.direction.clone(), ray.time);
            Some((
                &self.inner,
                1, // Increment count to avoid infinite recursion
            ))
        } else if count == 1 {
            if let Some(rec) = hit_record {
                rec.p += &self.offset;
            }
            *ray = Ray::new(&ray.origin + &self.offset, ray.direction.clone(), ray.time);
            None
        } else {
            unreachable!()
        }
    }
}

impl IntoBoundingBox for Translate<'_> {
    fn boundingbox(&self) -> BoundingBox {
        let inner_box = self.inner.boundingbox();
        inner_box.translate(&self.offset)
    }
}
