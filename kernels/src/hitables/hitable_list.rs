use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{
        HitKind, HitRecord, HitableWithCentroid, HitableWithCentroidDevice, RecursiveHitable,
    },
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
    hitables: SliceBuilder<'a, HitableWithCentroid<'a>>,
    bounding_box: BoundingBox,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> HitableList<'a> {
    pub fn new(hitables: Vec<HitKind<'a>>) -> Self {
        if hitables.len() > 32 {
            println!(
                "Warning: Creating HitableList with {} hitables. Consider using a BVH with more levels for better performance.",
                hitables.len()
            );
        }
        let bounding_box = hitables.iter().fold(BoundingBox::empty(), |acc, hitable| {
            acc.merge(&hitable.boundingbox())
        });
        let hitables = hitables.into_iter().map(|hitable| hitable.into()).collect();
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
        if (!self.bounding_box.hit(ray, interval)) || count >= self.hitables.len() {
            return None;
        }
        Some((&self.hitables.as_slice()[count].hitable, count + 1))
        // let min_distance_squared = if count == 0 {
        //     0.0
        // } else {
        //     (&ray.origin - &self.hitables[count - 1].centroid).length_squared()
        // };
        // let mut closest_distance_squared = Real::MAX;
        // let mut closest_index = 0;
        // for index in 0..self.hitables.len() {
        //     let distance_squared = (&ray.origin - &self.hitables[index].centroid).length_squared();
        //     if distance_squared < closest_distance_squared
        //         && (distance_squared > min_distance_squared
        //             || (distance_squared == min_distance_squared && index >= count))
        //     {
        //         closest_distance_squared = distance_squared;
        //         closest_index = index;
        //     }
        // }
        // if closest_distance_squared == Real::MAX {
        //     return None;
        // }

        // Some((
        //     &self.hitables.as_slice()[closest_index].hitable,
        //     closest_index + 1,
        // ))
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
            Some((&self.hitables[count].hitable, count + 1))
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
            Some((&self.hitables[index].hitable, 1))
        } else if count == 1 {
            None
        } else {
            unreachable!()
        }
    }

    #[cfg(not(target_os = "cuda"))]
    fn get_lights_inner<'a>(&'a self) -> Vec<HitKind<'a>>
    where
        Self: Into<HitKind<'a>>,
    {
        self.hitables
            .iter()
            .flat_map(|hitable_with_centroid| hitable_with_centroid.hitable.get_lights_inner())
            .collect()
    }
}
