use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, RecursiveHitable},
    random::Random,
    ray::Ray,
    stack::Stack,
    vec3::{Point3, Real, Vec3},
};
use gpu_builder::derive_builder;
use ref_builder::{RefBuilder, RefBuilderDevice};

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
pub struct Translate<'a> {
    offset: Vec3,
    inner: RefBuilder<'a, HitKind<'a>>,
    bounding_box: BoundingBox,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> Translate<'a> {
    pub fn new(offset: Vec3, inner: &'a HitKind<'a>) -> HitKind<'a> {
        let bounding_box = Self::make_boundingbox(inner, &offset);
        Translate {
            offset,
            inner: RefBuilder::new(inner),
            bounding_box,
        }
        .into()
    }

    pub fn new_owned(offset: Vec3, inner: HitKind<'a>) -> HitKind<'a> {
        let bounding_box = Self::make_boundingbox(&inner, &offset);
        Translate {
            offset,
            inner: RefBuilder::new_owned(inner),
            bounding_box,
        }
        .into()
    }

    fn make_boundingbox(inner: &HitKind<'a>, offset: &Vec3) -> BoundingBox {
        let inner_box = inner.boundingbox();
        inner_box.translate(offset)
    }
}

impl RecursiveHitable for Translate<'_> {
    fn hit_recursive<'a>(
        &'a self,
        ray: &mut Ray,
        _range: &mut Range<Real>,
        hit_record: &mut Option<HitRecord<'a>>,
        count: usize,
        _rng: &mut Random,
        _extra_stack: &mut Stack,
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

    fn pdf_value_recursive<'a>(
        &'a self,
        count: usize,
        origin: &mut Point3,
        _direction: &mut Vec3,
        _current_value: &mut Real,
        _rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count == 0 {
            *origin -= &self.offset;
            Some((&self.inner, 1))
        } else if count == 1 {
            *origin += &self.offset;
            None
        } else {
            unreachable!()
        }
    }

    fn random_recursive<'a>(
        &'a self,
        count: usize,
        origin: &mut Point3,
        _current_value: &mut Vec3,
        _rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count == 0 {
            *origin -= &self.offset;
            Some((&self.inner, 1))
        } else if count == 1 {
            *origin += &self.offset;
            None
        } else {
            unreachable!()
        }
    }
}

impl IntoBoundingBox for Translate<'_> {
    fn boundingbox(&self) -> BoundingBox {
        self.bounding_box.clone()
    }
}

#[cfg(not(target_os = "cuda"))]
use crate::hitables::GetLights;
#[cfg(not(target_os = "cuda"))]
use crate::hitables::hitable_list::HitableList;

#[cfg(not(target_os = "cuda"))]
impl<'a> GetLights<'a> for Translate<'a> {
    fn get_lights_inner(&self) -> Vec<HitKind<'a>> {
        let result = self.inner.get_lights_inner();
        if result.is_empty() {
            vec![]
        } else {
            vec![Translate::new_owned(
                self.offset.clone(),
                HitableList::new(result).into(),
            )]
        }
    }
}
