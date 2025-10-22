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
pub struct Scale<'a> {
    scale: Vec3,
    inner: RefBuilder<'a, HitKind<'a>>,
    bounding_box: BoundingBox,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> Scale<'a> {
    pub fn new(scale: Vec3, inner: &'a HitKind<'a>) -> HitKind<'a> {
        let bounding_box = Self::make_boundingbox(inner, &scale);
        Scale {
            scale,
            inner: RefBuilder::new(inner),
            bounding_box,
        }
        .into()
    }

    pub fn new_owned(scale: Vec3, inner: HitKind<'a>) -> HitKind<'a> {
        let bounding_box = Self::make_boundingbox(&inner, &scale);
        Scale {
            scale,
            inner: RefBuilder::new_owned(inner),
            bounding_box,
        }
        .into()
    }

    pub fn new_same(scale: Real, inner: &'a HitKind<'a>) -> HitKind<'a> {
        Self::new(Vec3::new(scale, scale, scale), inner)
    }

    pub fn new_owned_same(scale: Real, inner: HitKind<'a>) -> HitKind<'a> {
        Self::new_owned(Vec3::new(scale, scale, scale), inner)
    }

    fn make_boundingbox(inner: &HitKind<'a>, scale: &Vec3) -> BoundingBox {
        let inner_box = inner.boundingbox();
        inner_box.scale(scale)
    }
}

impl RecursiveHitable for Scale<'_> {
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
                rec.p = rec.p.scale_inverse(&self.scale);
                rec.normal = rec.normal.scale(&self.scale).normalize();
            }
            *ray = Ray::new(
                ray.origin.scale_inverse(&self.scale),
                ray.direction.scale_inverse(&self.scale),
                ray.time,
            );
            Some((
                &self.inner,
                1, // Increment count to avoid infinite recursion
            ))
        } else if count == 1 {
            if let Some(rec) = hit_record {
                rec.p = rec.p.scale(&self.scale);
                //normals are covariant
                rec.normal = rec.normal.scale_inverse(&self.scale).normalize();
            }
            *ray = Ray::new(
                ray.origin.scale(&self.scale),
                ray.direction.scale(&self.scale),
                ray.time,
            );
            None
        } else {
            unreachable!()
        }
    }

    fn pdf_value_recursive<'a>(
        &'a self,
        count: usize,
        origin: &mut Point3,
        direction: &mut Vec3,
        _current_value: &mut Real,
        _rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count == 0 {
            *origin = origin.scale_inverse(&self.scale);
            *direction = direction.scale_inverse(&self.scale);
            Some((&self.inner, 1))
        } else if count == 1 {
            *origin = origin.scale(&self.scale);
            *direction = direction.scale(&self.scale);
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
            *origin = origin.scale_inverse(&self.scale);
            Some((&self.inner, 1))
        } else if count == 1 {
            *origin = origin.scale(&self.scale);
            None
        } else {
            unreachable!()
        }
    }
}

impl IntoBoundingBox for Scale<'_> {
    fn boundingbox(&self) -> BoundingBox {
        self.bounding_box.clone()
    }
}

#[cfg(not(target_os = "cuda"))]
use crate::hitables::GetLights;
#[cfg(not(target_os = "cuda"))]
use crate::hitables::hitable_list::HitableList;

#[cfg(not(target_os = "cuda"))]
impl<'a> GetLights<'a> for Scale<'a> {
    fn get_lights_inner(&self) -> Vec<HitKind<'a>> {
        let result = self.inner.get_lights_inner();
        if result.is_empty() {
            vec![]
        } else {
            vec![Scale::new_owned(
                self.scale.clone(),
                HitableList::new_owned(result).into(),
            )]
        }
    }
}
