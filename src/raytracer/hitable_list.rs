use crate::raytracer::{
    hitable::{HitRecord, Hitable},
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
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_t = t_max;

        for hitable in &self.hitables {
            if let Some(hit_record) = hitable.hit(ray, t_min, closest_t) {
                closest_t = hit_record.t;
                closest_hit = Some(hit_record);
            }
        }

        closest_hit
    }
}
