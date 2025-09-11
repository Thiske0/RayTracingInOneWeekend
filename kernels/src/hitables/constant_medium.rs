use core::ops::Range;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{HitKind, HitKindDevice, HitRecord, RecursiveHitable},
    materials::{MaterialKind, MaterialKindDevice},
    random::{Random, RandomRange},
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};
use core::cell::Cell;
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
    #[no_copy]
    prev_hitrec: Cell<Option<HitRecord<'static>>>,
    #[no_copy]
    prev_ray_origin: Cell<Option<Point3>>,
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
            prev_hitrec: Cell::new(None),
            prev_ray_origin: Cell::new(None),
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
            prev_hitrec: Cell::new(None),
            prev_ray_origin: Cell::new(None),
        }
        .into()
    }
}

impl ConstantMedium<'_> {
    fn make_record<'a>(
        &'a self,
        ray: &mut Ray,
        start_t: Real,
        end_t: Real,
        rng: &mut Random,
    ) -> Option<HitRecord<'a>> {
        //first modify ray back to original
        let prev_record = self.prev_hitrec.take();
        *ray = Ray::new(
            self.prev_ray_origin.take().unwrap(),
            ray.direction.clone(),
            ray.time,
        );

        let distance_inside_boundary = (end_t - start_t) * ray.direction.length();
        let random_real: Real = rng.random_range(0.0..1.0);
        let hit_distance = self.neg_inv_density * random_real.ln();
        if hit_distance > distance_inside_boundary {
            return prev_record;
        }
        let t = start_t + hit_distance;
        if let Some(prev_record) = prev_record {
            if prev_record.t < t {
                return Some(prev_record);
            }
        }
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
    ) -> Option<(&'a HitKind<'a>, usize)> {
        if count == 0 {
            let static_hitrec = unsafe {
                core::mem::transmute::<Option<HitRecord<'a>>, Option<HitRecord<'static>>>(
                    hit_record.take(),
                )
            };
            self.prev_hitrec.set(static_hitrec);
            self.prev_ray_origin.set(Some(ray.origin.clone()));

            Some((
                &self.boundary,
                1, // Increment count to avoid infinite recursion
            ))
        } else if count == 1 {
            if let Some(boundary_hit) = hit_record {
                if boundary_hit.is_front_face {
                    // get second hit for back face
                    *ray = Ray::new(boundary_hit.p.clone(), ray.direction.clone(), ray.time);
                    *hit_record = None;
                    return Some((
                        &self.boundary,
                        2, // Increment count to avoid infinite recursion
                    ));
                } else {
                    // start is just the start of the ray
                    let start_t = range.start;
                    let end_t = boundary_hit.t;
                    //TODO: make work with concave shapes
                    *hit_record = self.make_record(ray, start_t, end_t, rng);
                    return None;
                }
            }
            *hit_record = self.make_record(ray, 0.0, 0.0, rng);
            None
        } else if count == 2 {
            if let Some(boundary_hit) = hit_record {
                // start is just the start of the ray
                let start_t = 0.0;
                let end_t = boundary_hit.t;
                *hit_record = self.make_record(ray, start_t, end_t, rng);
                return None;
            }
            let start_t = 0.0;
            let end_t = range.end;
            *hit_record = self.make_record(ray, start_t, end_t, rng);
            None
        } else {
            unreachable!()
        }
    }
}

impl IntoBoundingBox for ConstantMedium<'_> {
    fn boundingbox(&self) -> BoundingBox {
        self.boundary.boundingbox()
    }
}
