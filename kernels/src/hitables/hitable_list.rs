use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, RecursiveHitable},
    random::Random,
    ray::Ray,
    stack::Stack,
    vec3::Real,
};
use gpu_builder::derive_builder;
use ref_builder::{SliceBuilder, SliceBuilderDevice};

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
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

    pub fn new_owned(hitables: Vec<HitKind<'a>>) -> Self {
        let bounding_box = hitables.iter().fold(BoundingBox::empty(), |acc, hitable| {
            acc.merge(&hitable.boundingbox())
        });
        HitableList {
            hitables: SliceBuilder::new_owned(hitables),
            bounding_box,
        }
    }
}

impl IntoBoundingBox for HitableList<'_> {
    fn boundingbox(&self) -> BoundingBox {
        BoundingBox::empty().merge(&self.bounding_box)
    }
}

impl RecursiveHitable for HitableList<'_> {
    fn hit_recursive<'a>(
        &'a self,
        ray: &mut Ray,
        interval: &mut Range<Real>,
        _hit_record: &mut Option<HitRecord<'a>>,
        count: usize,
        _rng: &mut Random,
        _extra_stack: &mut Stack,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if !self.bounding_box.hit(ray, interval) {
            return None;
        }
        if count >= self.hitables.len() {
            return None;
        }
        Some((&self.hitables[count], count + 1))
    }
}
