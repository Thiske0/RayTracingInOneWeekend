use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, RecursiveHitable},
    random::Random,
    ray::Ray,
    stack::Stack,
    vec3::{Axis, Real},
};
use gpu_builder::derive_builder;
use ref_builder::{RefBuilder, RefBuilderDevice};

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
pub struct Rotate<'a> {
    axis: Axis,
    angle_rad: Real,
    inner: RefBuilder<'a, HitKind<'a>>,
    bounding_box: BoundingBox,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> Rotate<'a> {
    pub fn new(axis: Axis, angle_rad: Real, inner: &'a HitKind<'a>) -> HitKind<'a> {
        Rotate {
            axis,
            angle_rad,
            inner: RefBuilder::new(inner),
            bounding_box: Self::make_bounding_box(inner, axis, angle_rad),
        }
        .into()
    }

    pub fn new_owned(axis: Axis, angle_rad: Real, inner: HitKind<'a>) -> HitKind<'a> {
        let inner = RefBuilder::new_owned(inner);
        let bounding_box = Self::make_bounding_box(inner.as_ref(), axis, angle_rad);
        Rotate {
            axis,
            angle_rad,
            inner,
            bounding_box,
        }
        .into()
    }

    fn make_bounding_box(inner: &HitKind, axis: Axis, angle_rad: Real) -> BoundingBox {
        let inner_box = inner.boundingbox();
        let corners = inner_box.corners();
        let rotated_corners = corners.map(|corner| corner.rotate(&axis, angle_rad));
        BoundingBox::from_corners(&rotated_corners)
    }
}

impl RecursiveHitable for Rotate<'_> {
    fn hit_recursive<'a>(
        &'a self,
        ray: &mut Ray,
        range: &mut Range<Real>,
        hit_record: &mut Option<HitRecord<'a>>,
        count: usize,
        _rng: &mut Random,
        _extra_stack: &mut Stack,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count == 0 {
            if !self.bounding_box.hit(ray, range) {
                return None;
            }

            if let Some(rec) = hit_record {
                rec.p = rec.p.rotate(&self.axis, -self.angle_rad);
                rec.normal = rec.normal.rotate(&self.axis, -self.angle_rad);
            }

            *ray = Ray::new(
                ray.origin.rotate(&self.axis, -self.angle_rad),
                ray.direction.rotate(&self.axis, -self.angle_rad),
                ray.time,
            );
            Some((&self.inner, 1))
        } else if count == 1 {
            if let Some(rec) = hit_record {
                rec.p = rec.p.rotate(&self.axis, self.angle_rad);
                rec.normal = rec.normal.rotate(&self.axis, self.angle_rad);
            }
            *ray = Ray::new(
                ray.origin.rotate(&self.axis, self.angle_rad),
                ray.direction.rotate(&self.axis, self.angle_rad),
                ray.time,
            );
            None
        } else {
            unreachable!()
        }
    }
}

impl IntoBoundingBox for Rotate<'_> {
    fn boundingbox(&self) -> BoundingBox {
        self.bounding_box.clone()
    }
}
