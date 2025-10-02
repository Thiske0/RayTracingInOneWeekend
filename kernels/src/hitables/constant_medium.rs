use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, RecursiveHitable},
    materials::{MaterialKind, MaterialKindDevice},
    random::{Random, RandomRange},
    ray::Ray,
    stack::Stack,
    vec3::{Real, Vec3},
};
use gpu_builder::derive_builder;
use ref_builder::{RefBuilder, RefBuilderDevice};

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[repr(C)]
#[derive_builder('a)]
pub struct ConstantMedium<'a> {
    neg_inv_density: Real,
    boundary: RefBuilder<'a, HitKind<'a>>,
    phase_function: MaterialKind<'a>,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> ConstantMedium<'a> {
    pub fn new(
        density: Real,
        boundary: &'a HitKind<'a>,
        phase_function: MaterialKind<'a>,
    ) -> HitKind<'a> {
        ConstantMedium {
            neg_inv_density: -1.0 / density,
            boundary: RefBuilder::new(boundary),
            phase_function,
        }
        .into()
    }

    pub fn new_owned(
        density: Real,
        boundary: HitKind<'a>,
        phase_function: MaterialKind<'a>,
    ) -> HitKind<'a> {
        ConstantMedium {
            neg_inv_density: -1.0 / density,
            boundary: RefBuilder::new_owned(boundary),
            phase_function,
        }
        .into()
    }
}

impl ConstantMedium<'_> {
    fn make_record<'a>(
        &'a self,
        ray: &Ray,
        start_t: Real,
        end_t: Real,
        rng: &mut Random,
    ) -> Option<HitRecord<'a>> {
        let distance_inside_boundary = (end_t - start_t) * ray.direction.length();
        let random_real: Real = rng.random_range(0.0..1.0);
        let hit_distance = self.neg_inv_density * random_real.ln();
        if hit_distance > distance_inside_boundary {
            return None;
        }
        let t = start_t + hit_distance / ray.direction.length();
        let normal = Vec3::new(1.0, 0.0, 0.0); // arbitrary
        let (u, v) = (0.0, 0.0); // arbitrary
        Some(HitRecord {
            t,
            p: ray.at(t),
            normal,
            mat: &self.phase_function,
            u,
            v,
            is_front_face: true,
        })
    }
}

impl RecursiveHitable for ConstantMedium<'_> {
    fn hit_recursive<'a>(
        &'a self,
        ray: &mut Ray,
        range: &mut Range<Real>,
        hit_record: &mut Option<HitRecord<'a>>,
        count: usize,
        rng: &mut Random,
        extra_stack: &mut Stack,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count == 0 {
            if !self.boundingbox().hit(ray, range) {
                return None;
            }

            extra_stack
                .push::<(Range<Real>, Option<HitRecord<'a>>)>((range.clone(), hit_record.take()));
            range.end = Real::INFINITY;
            return Some((&self.boundary, 1));
        } else if count == 1 {
            if let Some(first_hit) = hit_record.take() {
                if first_hit.is_front_face {
                    // get second hit for back face
                    *range = (1e-3 + first_hit.t)..Real::INFINITY;
                    return Some((&self.boundary, 2));
                } else {
                    *hit_record = self.make_record(&ray, range.start, first_hit.t, rng);
                }
            }
        } else if count == 2 {
            let start_t = range.start - 1e-3;
            let end_t = if let Some(second_hit) = hit_record.take() {
                second_hit.t
            } else {
                // no second hit, this means we are really close to the boundary
                start_t
            };
            *hit_record = self.make_record(&ray, start_t, end_t, rng);
        } else {
            unreachable!()
        }

        // restore previous hit_record and range
        let (prev_range, prev_hit_record) =
            unsafe { extra_stack.pop::<(Range<Real>, Option<HitRecord<'a>>)>() };
        let mut reset_record = false;
        if let Some(ref prev_rec) = prev_hit_record {
            if let Some(rec) = hit_record {
                // merge with previous hit record
                if rec.t > prev_rec.t {
                    reset_record = true;
                }
            } else {
                reset_record = true;
            }
        }
        if reset_record {
            *hit_record = prev_hit_record;
            *range = prev_range;
        } else if let Some(hit_record) = hit_record {
            *range = prev_range.start..hit_record.t;
        }
        None
    }
}

impl IntoBoundingBox for ConstantMedium<'_> {
    fn boundingbox(&self) -> BoundingBox {
        self.boundary.boundingbox()
    }
}
