use std::ops::Range;

use crate::raytracer::{
    hitable::{HitKind, HitRecord, Hitable},
    ray::Ray,
};

use simple_ray_tracer_kernels::vec3::Real;

pub struct HitableListBuilder<'a> {
    hitables: Vec<HitKind<'a>>,
}

pub struct HitableList<'a> {
    hitables: &'a [HitKind<'a>],
}

impl<'a> HitableListBuilder<'a> {
    pub fn new() -> Self {
        HitableListBuilder {
            hitables: Vec::new(),
        }
    }

    pub fn add(&mut self, hitable: HitKind<'a>) {
        self.hitables.push(hitable);
    }

    pub fn build(&'a mut self) -> HitKind<'a> {
        HitableList {
            hitables: self.hitables.as_slice(),
        }
        .into()
    }
}

impl Hitable for HitableList<'_> {
    fn hit<'a>(&'a self, ray: &Ray, interval: &Range<Real>) -> Option<HitRecord<'a>> {
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
