use core::ops::Range;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
use enum_dispatch::enum_dispatch;

use crate::{
    ray::Ray,
    vec3::{Axis, Point3, Real},
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

        for axis in Axis::items() {
            let inv_d = 1.0 / (&ray.direction)[&axis];
            let mut t0 = ((&self.min)[&axis] - (&ray.origin)[&axis]) * inv_d;
            let mut t1 = ((&self.max)[&axis] - (&ray.origin)[axis]) * inv_d;

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

    pub fn longest_axis(&self) -> Axis {
        let dx = self.max.x - self.min.x;
        let dy = self.max.y - self.min.y;
        let dz = self.max.z - self.min.z;

        if dx >= dy && dx >= dz {
            Axis::X
        } else if dy >= dz {
            Axis::Y
        } else {
            Axis::Z
        }
    }

    pub fn center(&self) -> Point3 {
        (&self.min + &self.max) / 2.0
    }
}

#[enum_dispatch]
pub trait IntoBoundingBox {
    fn boundingbox(&self) -> BoundingBox;
}
