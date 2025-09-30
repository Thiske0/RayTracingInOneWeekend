use core::{ops::Range, panic};

use enum_dispatch::enum_dispatch;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{
        constant_medium::{ConstantMedium, ConstantMediumDevice},
        hitable_list::{HitableList, HitableListDevice},
        planar::{Quad, QuadDevice, Triangle, TriangleDevice},
        rotate::{Rotate, RotateDevice},
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
pub mod sphere;
pub mod translate;

#[cfg(not(target_os = "cuda"))]
pub mod hitable_list_builder;

#[enum_dispatch]
pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>, rng: &mut Random)
    -> Option<HitRecord<'a>>;
}

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
}

#[repr(C)]
#[derive_builder('b)]
#[enum_dispatch(IntoBoundingBox)]
pub enum HitKind<'b> {
    HitKindNonRecursive(HitKindNonRecursive<'b>),
    HitKindRecursive(HitKindRecursive<'b>),
}

pub const STACK_SIZE: usize = 16;

pub struct StackEntry<'a, 'b> {
    pub hitkind: &'a HitKindRecursive<'b>,
    pub next_count: usize,
}

impl<'a, 'b> StackEntry<'a, 'b> {
    pub fn new(hitkind: &'a HitKindRecursive<'b>) -> Self {
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
    if let HitKind::HitKindRecursive(hitkind) = hitkind {
        stack[0] = StackEntry::new(hitkind);
    }
    stack
}

fn init_stack_for_recursive<'a, 'b, const SIZE: usize>(
    hitkind: &'a HitKindRecursive<'b>,
) -> [StackEntry<'a, 'b>; SIZE] {
    let mut stack: [StackEntry<'a, 'b>; SIZE] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    stack[0] = StackEntry::new(hitkind);
    stack
}

impl HitKind<'_> {
    pub fn hit<'a>(
        &'a self,
        ray: Ray,
        range: Range<Real>,
        rng: &mut Random,
    ) -> Option<HitRecord<'a>> {
        match self {
            HitKind::HitKindNonRecursive(h) => h.hit(&ray, &range, rng),
            HitKind::HitKindRecursive(h) => Self::hit_recursive(h, ray, range, rng),
        }
    }

    fn hit_recursive<'a>(
        hitkind: &'a HitKindRecursive<'a>,
        ray: Ray,
        range: Range<Real>,
        rng: &mut Random,
    ) -> Option<HitRecord<'a>> {
        let mut extra_stack = Stack::new();
        let mut stack = init_stack_for_recursive::<STACK_SIZE>(hitkind);
        let mut stack_ptr = 1;

        let mut hit_record: Option<HitRecord<'a>> = None;
        let mut ray = ray;
        let mut range = range;
        loop {
            if stack_ptr >= STACK_SIZE {
                // Stack overflow, break to avoid infinite loop
                panic!("Stack overflow in hit_recursive");
            }
            let current = &mut stack[stack_ptr - 1];
            if let Some((inner_hitkind, next_count)) = current.hitkind.hit_recursive(
                &mut ray,
                &mut range,
                &mut hit_record,
                current.next_count,
                rng,
                &mut extra_stack,
            ) {
                current.next_count = next_count;
                match inner_hitkind {
                    HitKind::HitKindNonRecursive(inner_hitkind) => {
                        if let Some(rec) = inner_hitkind.hit(&ray, &range, rng) {
                            if let Some(current_rec) = &hit_record {
                                if rec.t < current_rec.t {
                                    range = range.start..rec.t;
                                    hit_record = Some(rec);
                                }
                            } else {
                                range = range.start..rec.t;
                                hit_record = Some(rec);
                            }
                        }
                    }
                    HitKind::HitKindRecursive(inner_hitkind) => {
                        stack_ptr += 1;
                        stack[stack_ptr - 1] = StackEntry::new(inner_hitkind);
                    }
                }
            } else {
                stack_ptr -= 1;
                if stack_ptr == 0 {
                    return hit_record;
                }
            }
        }
    }
}

// Macro's
macro_rules! impl_into_hitkind_recursive {
    ($($type:ident),*) => {
        $(
            impl<'a> From<$type<'a>> for HitKind<'a> {
                fn from(value: $type<'a>) -> Self {
                    HitKind::HitKindRecursive(value.into())
                }
            }
        )*
    };
}
macro_rules! impl_into_hitkind_non_recursive {
    ($($type:ident),*) => {
        $(
            impl<'a> From<$type<'a>> for HitKind<'a> {
                fn from(value: $type<'a>) -> Self {
                    HitKind::HitKindNonRecursive(value.into())
                }
            }
        )*
    };
}

#[repr(C)]
#[enum_dispatch(Hitable, IntoBoundingBox)]
#[derive_builder('b)]
pub enum HitKindNonRecursive<'b> {
    Sphere(Sphere<'b>),
    Quad(Quad<'b>),
    Triangle(Triangle<'b>),
}
impl_into_hitkind_non_recursive!(Sphere, Quad, Triangle);

#[repr(C)]
#[enum_dispatch(IntoBoundingBox)]
#[derive_builder('b)]
pub enum HitKindRecursive<'b> {
    HitableList(HitableList<'b>),
    Translate(Translate<'b>),
    Rotate(Rotate<'b>),
    ConstantMedium(ConstantMedium<'b>),
}

impl_into_hitkind_recursive!(HitableList, Translate, Rotate, ConstantMedium);

impl RecursiveHitable for HitKindRecursive<'_> {
    fn hit_recursive<'a>(
        &'a self,
        ray: &mut Ray,
        range: &mut Range<Real>,
        hit_record: &mut Option<HitRecord<'a>>,
        count: usize,
        rng: &mut Random,
        extra_stack: &mut Stack,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        match self {
            HitKindRecursive::HitableList(h) => {
                h.hit_recursive(ray, range, hit_record, count, rng, extra_stack)
            }
            HitKindRecursive::Translate(h) => {
                h.hit_recursive(ray, range, hit_record, count, rng, extra_stack)
            }
            HitKindRecursive::Rotate(h) => {
                h.hit_recursive(ray, range, hit_record, count, rng, extra_stack)
            }
            HitKindRecursive::ConstantMedium(h) => {
                h.hit_recursive(ray, range, hit_record, count, rng, extra_stack)
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
