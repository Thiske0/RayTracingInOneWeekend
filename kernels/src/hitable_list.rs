use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitable::{HitKind, HitRecord, Hitable},
    ray::Ray,
    vec3::Real,
};

#[cfg(not(target_os = "cuda"))]
use cust::memory::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy))]
pub struct HitableList<'a> {
    hitables: &'a [HitKind<'a>],
    bounding_box: BoundingBox,
}
#[cfg(not(target_os = "cuda"))]
unsafe impl<'a> DeviceCopy for HitableList<'a> {}

impl<'a> HitableList<'a> {
    pub fn new(hitables: &'a [HitKind<'a>], bounding_box: BoundingBox) -> Self {
        HitableList {
            hitables,
            bounding_box,
        }
    }
}

impl IntoBoundingBox for HitableList<'_> {
    fn boundingbox(&self) -> BoundingBox {
        BoundingBox::empty().merge(&self.bounding_box)
    }
}

impl Hitable for HitableList<'_> {
    fn hit<'a>(&'a self, ray: &Ray, interval: &Range<Real>) -> Option<HitRecord<'a>> {
        if !self.bounding_box.hit(ray, interval) {
            return None;
        }

        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_interval = interval.clone();

        for hitable in self.hitables {
            if let Some(hit_record) = hitable.hit(ray, &closest_interval) {
                closest_interval = closest_interval.start..hit_record.t;
                closest_hit = Some(hit_record);
            }
        }

        closest_hit
    }
}
