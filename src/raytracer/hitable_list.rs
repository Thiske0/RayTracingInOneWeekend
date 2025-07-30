use crate::raytracer::{
    hitable::{HitRecord, Hitable},
    interval::Interval,
    ray::Ray,
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
    fn hit(&self, ray: &Ray, interval: &Interval) -> Option<HitRecord> {
        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_interval = interval.clone();

        for hitable in &self.hitables {
            if let Some(hit_record) = hitable.hit(ray, &closest_interval) {
                closest_interval = Interval::new(closest_interval.min, hit_record.t);
                closest_hit = Some(hit_record);
            }
        }

        closest_hit
    }
}
