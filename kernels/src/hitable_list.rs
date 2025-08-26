use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitable::{HitKind, HitKindDevice, HitRecord, Hitable},
    ray::Ray,
    vec3::Real,
};
use gpu_builder::Builder;
use ref_builder::{SliceBuilder, SliceBuilderDevice};

#[repr(C)]
#[derive(Builder)]
#[use_lifetime("'a")]
pub struct HitableList<'a> {
    hitables: SliceBuilder<'a, HitKind<'a>>,
    bounding_box: BoundingBox,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> HitableList<'a> {
    pub fn new(hitables: &'a [HitKind<'a>]) -> Self {
        let bounding_box = hitables.iter().fold(BoundingBox::empty(), |acc, hitable| {
            acc.merge(&hitable.boundingbox())
        });
        HitableList {
            hitables: SliceBuilder::new(hitables),
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

        for hitable in self.hitables.as_slice() {
            if let Some(hit_record) = hitable.hit(ray, &closest_interval) {
                closest_interval = closest_interval.start..hit_record.t;
                closest_hit = Some(hit_record);
            }
        }

        closest_hit
    }
}
