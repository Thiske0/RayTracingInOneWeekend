use core::ops::Range;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

use crate::{
    ray::Ray,
    vec3::{Point3, Real},
};

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy, Debug))]
pub struct BoundingBox {
    pub min: Point3,
    pub max: Point3,
}

impl BoundingBox {
    pub fn new(min: Point3, max: Point3) -> Self {
        BoundingBox { min, max }
    }

    pub fn empty() -> Self {
        BoundingBox {
            min: Point3::new(Real::MAX, Real::MAX, Real::MAX),
            max: Point3::new(Real::MIN, Real::MIN, Real::MIN),
        }
    }

    pub fn merge(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn hit(&self, ray: &Ray, range: &Range<Real>) -> bool {
        let mut t_min = range.start;
        let mut t_max = range.end;

        for i in 0..3 {
            let inv_d = 1.0 / (&ray.direction)[i];
            let mut t0 = ((&self.min)[i] - (&ray.origin)[i]) * inv_d;
            let mut t1 = ((&self.max)[i] - (&ray.origin)[i]) * inv_d;

            if inv_d < 0.0 {
                core::mem::swap(&mut t0, &mut t1);
            }

            t_min = t_min.max(t0);
            t_max = t_max.min(t1);

            if t_max <= t_min {
                return false;
            }
        }

        true
    }
}

pub trait IntoBoundingBox {
    fn boundingbox(&self) -> BoundingBox;
}
