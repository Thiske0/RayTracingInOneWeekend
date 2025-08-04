use std::ops::Range;

use crate::raytracer::{
    hitable::{HitKind, HitRecord, Hitable},
    ray::Ray,
    vec3::Real,
};

pub struct HitableList {
    hitables: Vec<HitKind>,
}

impl HitableList {
    pub fn new() -> Self {
        HitableList {
            hitables: Vec::new(),
        }
    }

    pub fn add(&mut self, hitable: HitKind) {
        self.hitables.push(hitable);
    }
}

impl Hitable for HitableList {
    fn hit<'a>(&'a self, ray: &Ray, interval: &Range<Real>) -> Option<HitRecord<'a>> {
        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_interval = interval.clone();

        for hitable in &self.hitables {
            if let Some(hit_record) = hitable.hit(ray, &closest_interval) {
                closest_interval = closest_interval.start..hit_record.t;
                closest_hit = Some(hit_record);
            }
        }

        closest_hit
    }
}
