use core::{ops::Range, panic};

use enum_dispatch::enum_dispatch;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{
        constant_medium::{ConstantMedium, ConstantMediumDevice},
        hitable_list::{HitableList, HitableListDevice},
        planar::{
            NormedTriangle, NormedTriangleDevice, Quad, QuadDevice, Triangle, TriangleDevice,
        },
        rotate::{Rotate, RotateDevice},
        scale::{Scale, ScaleDevice},
        sphere::{Sphere, SphereDevice},
        translate::{Translate, TranslateDevice},
    },
    materials::MaterialKind,
    random::Random,
    ray::Ray,
    stack::Stack,
    vec3::{Point3, Real, Vec3},
};
use gpu_builder::derive_builder;

pub mod constant_medium;
pub mod hitable_list;
pub mod planar;
pub mod rotate;
pub mod scale;
pub mod sphere;
pub mod translate;

#[cfg(not(target_os = "cuda"))]
pub mod hitable_list_builder;
#[cfg(not(target_os = "cuda"))]
pub mod object_parser;

#[enum_dispatch]
pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>, rng: &mut Random)
    -> Option<HitRecord<'a>>;
    fn pdf_value(&self, origin: &Point3, direction: &Vec3, rng: &mut Random) -> Real;
    fn random(&self, origin: &Point3, rng: &mut Random) -> Vec3;
    #[cfg(not(target_os = "cuda"))]
    fn is_light(&self) -> bool;
}

#[cfg(not(target_os = "cuda"))]
pub trait FullHitable: Hitable + Clone {}
#[cfg(target_os = "cuda")]
pub trait FullHitable: Hitable {}

#[cfg(not(target_os = "cuda"))]
impl<T: Hitable + Clone> FullHitable for T {}

#[cfg(target_os = "cuda")]
impl<T: Hitable> FullHitable for T {}

#[enum_dispatch]
pub trait RecursiveHitable {
    //gets a ray and a range and a mutable hit_record
    //a usize specifies how many times we have passed this RecursiveHitable already
    //if intersection occurs, update hit_record
    //returns a HitKind to recurse into if needed also returns a usize that indicates where to continue
    //also specifies a Ray and Range for the recursive call
    fn hit_recursive<'a>(
        &'a self,
        ray: &mut Ray,
        range: &mut Range<Real>,
        hit_record: &mut Option<HitRecord<'a>>,
        count: usize,
        rng: &mut Random,
        extra_stack: &mut Stack,
    ) -> Option<(&'a HitKind<'a>, usize)>;
    fn pdf_value_recursive<'a>(
        &'a self,
        count: usize,
        origin: &mut Point3,
        direction: &mut Vec3,
        current_value: &mut Real,
        rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)>;
    fn random_recursive<'a>(
        &'a self,
        count: usize,
        origin: &mut Point3,
        current_value: &mut Vec3,
        rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)>;

    #[cfg(not(target_os = "cuda"))]
    fn get_lights_inner<'a>(&'a self) -> Vec<HitKind<'a>>
    where
        Self: Into<HitKind<'a>>;
}

impl<T: FullHitable> RecursiveHitable for T {
    fn hit_recursive<'b>(
        &'b self,
        ray: &mut Ray,
        range: &mut Range<Real>,
        hit_record: &mut Option<HitRecord<'b>>,
        _count: usize,
        rng: &mut Random,
        _extra_stack: &mut Stack,
    ) -> Option<(&'b HitKind<'b>, usize)> {
        if let Some(rec) = self.hit(ray, range, rng) {
            if let Some(current_rec) = hit_record {
                if rec.t < current_rec.t {
                    *range = range.start..rec.t;
                    *hit_record = Some(rec);
                }
            } else {
                *range = range.start..rec.t;
                *hit_record = Some(rec);
            }
        }
        None
    }

    fn pdf_value_recursive<'b>(
        &'b self,
        _count: usize,
        origin: &mut Point3,
        direction: &mut Vec3,
        current_value: &mut Real,
        rng: &mut Random,
    ) -> Option<(&'b HitKind<'b>, usize)> {
        let value = self.pdf_value(origin, direction, rng);
        *current_value += value;
        None
    }

    fn random_recursive<'b>(
        &'b self,
        _count: usize,
        origin: &mut Point3,
        current_value: &mut Vec3,
        rng: &mut Random,
    ) -> Option<(&'b HitKind<'b>, usize)> {
        let value = self.random(origin, rng);
        *current_value += value;
        None
    }

    #[cfg(not(target_os = "cuda"))]
    fn get_lights_inner<'b>(&self) -> Vec<HitKind<'b>>
    where
        Self: Into<HitKind<'b>>,
    {
        if self.is_light() {
            vec![self.clone().into()]
        } else {
            vec![]
        }
    }
}

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('b)]
#[enum_dispatch(IntoBoundingBox, RecursiveHitable)]
pub enum HitKind<'b> {
    HitableList(HitableList<'b>),
    Translate(Translate<'b>),
    Rotate(Rotate<'b>),
    Scale(Scale<'b>),
    ConstantMedium(ConstantMedium<'b>),
    Sphere(Sphere<'b>),
    Quad(Quad<'b>),
    Triangle(Triangle<'b>),
    NormedTriangle(NormedTriangle<'b>),
}

pub const STACK_SIZE: usize = 32;

pub struct StackEntry<'a, 'b> {
    pub hitkind: &'a HitKind<'b>,
    pub next_count: usize,
}

impl<'a, 'b> StackEntry<'a, 'b> {
    pub fn new(hitkind: &'a HitKind<'b>) -> Self {
        StackEntry {
            hitkind,
            next_count: 0,
        }
    }
}

pub fn init_stack<'a, 'b, const SIZE: usize>(
    hitkind: &'a HitKind<'b>,
) -> [StackEntry<'a, 'b>; SIZE] {
    let mut stack: [StackEntry<'a, 'b>; SIZE] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    stack[0] = StackEntry::new(hitkind);
    stack
}

fn init_stack_for_recursive<'a, 'b, const SIZE: usize>(
    hitkind: &'a HitKind<'b>,
) -> [StackEntry<'a, 'b>; SIZE] {
    let mut stack: [StackEntry<'a, 'b>; SIZE] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    stack[0] = StackEntry::new(hitkind);
    stack
}

trait Recursive<'a> {
    fn recurse(
        &mut self,
        hitkind: &'a HitKind<'a>,
        count: usize,
    ) -> Option<(&'a HitKind<'a>, usize)>;
}

struct HitRecursive<'a, 'b> {
    ray: Ray,
    range: Range<Real>,
    hit_record: Option<HitRecord<'a>>,
    rng: &'b mut Random,
    extra_stack: Stack,
}

impl<'a> Recursive<'a> for HitRecursive<'a, '_> {
    fn recurse(
        &mut self,
        hitkind: &'a HitKind<'a>,
        count: usize,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        hitkind.hit_recursive(
            &mut self.ray,
            &mut self.range,
            &mut self.hit_record,
            count,
            &mut self.rng,
            &mut self.extra_stack,
        )
    }
}

struct PDFValueRecursive<'b> {
    origin: Point3,
    direction: Vec3,
    current_value: Real,
    rng: &'b mut Random,
}

impl<'a> Recursive<'a> for PDFValueRecursive<'_> {
    fn recurse(
        &mut self,
        hitkind: &'a HitKind<'a>,
        count: usize,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        hitkind.pdf_value_recursive(
            count,
            &mut self.origin,
            &mut self.direction,
            &mut self.current_value,
            self.rng,
        )
    }
}

struct PDFRandomRecursive<'b> {
    origin: Point3,
    current_value: Vec3,
    rng: &'b mut Random,
}

impl<'a> Recursive<'a> for PDFRandomRecursive<'_> {
    fn recurse(
        &mut self,
        hitkind: &'a HitKind<'a>,
        count: usize,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        hitkind.random_recursive(count, &mut self.origin, &mut self.current_value, self.rng)
    }
}

impl<'b> HitKind<'b> {
    pub fn hit<'a>(
        &'a self,
        ray: Ray,
        range: Range<Real>,
        rng: &mut Random,
    ) -> Option<HitRecord<'a>> {
        let mut recurse_impl = HitRecursive {
            ray: ray,
            range: range,
            hit_record: None,
            rng: rng,
            extra_stack: Stack::new(),
        };

        Self::use_stack(self, &mut recurse_impl);
        return recurse_impl.hit_record;
    }

    pub fn pdf_value(&self, origin: &Point3, direction: &Vec3, rng: &mut Random) -> Real {
        let mut recurse_impl = PDFValueRecursive {
            origin: origin.clone(),
            direction: direction.clone(),
            current_value: 0.0,
            rng: rng,
        };

        Self::use_stack(self, &mut recurse_impl);
        return recurse_impl.current_value;
    }

    pub fn random(&self, origin: &Point3, rng: &mut Random) -> Vec3 {
        let mut recurse_impl = PDFRandomRecursive {
            origin: origin.clone(),
            current_value: Vec3::zero(),
            rng: rng,
        };

        Self::use_stack(self, &mut recurse_impl);
        return recurse_impl.current_value;
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn get_lights(&'b self) -> HitKind<'b> {
        let lights = self.get_lights_inner();
        HitableList::new(lights).into()
    }

    fn use_stack<'a, RecursiveImpl: Recursive<'a>>(
        initial: &'a HitKind<'a>,
        recurse_impl: &mut RecursiveImpl,
    ) {
        let mut stack = init_stack_for_recursive::<STACK_SIZE>(initial);
        let mut stack_ptr = 1;

        loop {
            if stack_ptr >= STACK_SIZE {
                // Stack overflow, break to avoid infinite loop
                panic!("Stack overflow in hit_recursive");
            }
            let current = &mut stack[stack_ptr - 1];
            if let Some((inner_hitkind, next_count)) =
                recurse_impl.recurse(current.hitkind, current.next_count)
            {
                current.next_count = next_count;
                stack_ptr += 1;
                stack[stack_ptr - 1] = StackEntry::new(inner_hitkind);
            } else {
                stack_ptr -= 1;
                if stack_ptr == 0 {
                    return;
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub t: Real,
    pub is_front_face: bool,
    pub mat: &'a MaterialKind<'a>,
    pub u: Real,
    pub v: Real,
}

impl<'a> HitRecord<'a> {
    pub fn new(
        p: Point3,
        normal: Vec3,
        t: Real,
        is_front_face: bool,
        mat: &'a MaterialKind,
        (u, v): (Real, Real),
    ) -> Self {
        HitRecord {
            p,
            normal,
            t,
            is_front_face,
            mat,
            u,
            v,
        }
    }
}
