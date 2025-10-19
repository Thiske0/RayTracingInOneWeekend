use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, RecursiveHitable},
    random::{Random, RandomRange},
    ray::Ray,
    stack::Stack,
    vec3::{Point3, Real, Vec3},
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

    fn pdf_value_recursive<'a>(
        &'a self,
        count: usize,
        _origin: &mut Point3,
        _direction: &mut Vec3,
        current_value: &mut Real,
        _rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count >= self.hitables.len() {
            *current_value /= self.hitables.len() as Real;
            None
        } else {
            Some((&self.hitables[count], count + 1))
        }
    }

    fn random_recursive<'a>(
        &'a self,
        count: usize,
        _origin: &mut Point3,
        _current_value: &mut Vec3,
        rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count == 0 {
            if self.hitables.len() == 0 {
                return None;
            }
            let index = rng.random_range(0..self.hitables.len());
            Some((&self.hitables[index], 1))
        } else if count == 1 {
            None
        } else {
            unreachable!()
        }
    }
}
#[cfg(not(target_os = "cuda"))]
use crate::hitables::GetLights;

#[cfg(not(target_os = "cuda"))]
impl<'a> GetLights<'a> for HitableList<'a> {
    fn get_lights_inner(&self) -> Vec<HitKind<'a>> {
        self.hitables
            .iter()
            .flat_map(|hitable| hitable.get_lights_inner())
            .collect()
    }
}
