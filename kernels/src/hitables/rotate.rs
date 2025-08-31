use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, Hitable},
    ray::Ray,
    vec3::{Axis, Real},
};
use gpu_builder::derive_builder;
use ref_builder::{RefBuilder, RefBuilderDevice};

#[repr(C)]
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

impl Hitable for Rotate<'_> {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>> {
        if !self.bounding_box.hit(ray, range) {
            return None;
        }

        let rotated_ray = Ray::new(
            ray.origin.rotate(&self.axis, -self.angle_rad),
            ray.direction.rotate(&self.axis, -self.angle_rad),
            ray.time,
        );
        if let Some(mut rec) = self.inner.hit(&rotated_ray, range) {
            rec.p = rec.p.rotate(&self.axis, self.angle_rad);
            rec.normal = rec.normal.rotate(&self.axis, self.angle_rad);
            return Some(rec);
        }
        None
    }
}

impl IntoBoundingBox for Rotate<'_> {
    fn boundingbox(&self) -> BoundingBox {
        let inner_box = self.inner.boundingbox();
        let corners = inner_box.corners();
        self.bounding_box.clone()
    }
}
