use std::ops::Range;

use crate::raytracer::{
    hitable::{HitRecord, Hitable},
    ray::Ray,
    vec3::Real,
};

pub struct HitableList<'a> {
    hitables: Vec<Box<dyn Hitable + 'a>>,
}

impl<'a> HitableList<'a> {
    pub fn new() -> Self {
        HitableList {
            hitables: Vec::new(),
        }
    }

    pub fn add<T: Hitable + 'a>(&mut self, hitable: T) {
        self.hitables.push(Box::new(hitable));
    }
}

impl<'a> Hitable for HitableList<'a> {
    fn hit<'b>(&'b self, ray: &Ray, interval: &Range<Real>) -> Option<HitRecord<'b>> {
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
