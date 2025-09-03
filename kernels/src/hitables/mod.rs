use core::{ops::Range, panic};

use enum_dispatch::enum_dispatch;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::{
        hitable_list::{HitableList, HitableListDevice},
        planar::{Quad, QuadDevice, Triangle, TriangleDevice},
        rotate::{Rotate, RotateDevice},
        sphere::{Sphere, SphereDevice},
        translate::{Translate, TranslateDevice},
    },
    materials::MaterialKind,
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};
use gpu_builder::derive_builder;

pub mod hitable_list;
pub mod planar;
pub mod rotate;
pub mod sphere;
pub mod translate;

#[cfg(not(target_os = "cuda"))]
pub mod hitable_list_builder;

#[enum_dispatch]
pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, range: &Range<Real>) -> Option<HitRecord<'a>>;
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
        range: &Range<Real>,
        hit_record: &mut Option<HitRecord<'a>>,
        count: usize,
    ) -> Option<(&'a HitKind<'a>, usize)>;
}

#[repr(C)]
#[derive_builder('b)]
#[enum_dispatch(IntoBoundingBox)]
pub enum HitKind<'b> {
    HitKindNonRecursive(HitKindNonRecursive<'b>),
    HitKindRecursive(HitKindRecursive<'b>),
}

const STACK_SIZE: usize = 16;

struct StackEntry<'a> {
    pub hitkind: &'a HitKindRecursive<'a>,
    pub next_count: usize,
}

impl<'a> StackEntry<'a> {
    pub fn new(hitkind: &'a HitKindRecursive<'a>) -> Self {
        StackEntry {
            hitkind,
            next_count: 0,
        }
    }
}

fn init_stack<'a, const SIZE: usize>(hitkind: &'a HitKindRecursive<'a>) -> [StackEntry<'a>; SIZE] {
    let mut stack: [StackEntry<'a>; SIZE] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    stack[0] = StackEntry::new(hitkind);
    stack
}

impl HitKind<'_> {
    pub fn hit<'a>(&'a self, ray: Ray, range: Range<Real>) -> Option<HitRecord<'a>> {
        match self {
            HitKind::HitKindNonRecursive(h) => h.hit(&ray, &range),
            HitKind::HitKindRecursive(h) => Self::hit_recursive(h, ray, range),
        }
    }

    fn hit_recursive<'a>(
        hitkind: &'a HitKindRecursive<'a>,
        ray: Ray,
        range: Range<Real>,
    ) -> Option<HitRecord<'a>> {
        let mut stack = init_stack::<STACK_SIZE>(hitkind);
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
            if let Some((inner_hitkind, next_count)) =
                current
                    .hitkind
                    .hit_recursive(&mut ray, &range, &mut hit_record, current.next_count)
            {
                current.next_count = next_count;
                match inner_hitkind {
                    HitKind::HitKindNonRecursive(inner_hitkind) => {
                        if let Some(rec) = inner_hitkind.hit(&ray, &range) {
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
#[derive_builder('b)]
#[enum_dispatch(Hitable, IntoBoundingBox)]
pub enum HitKindNonRecursive<'b> {
    Sphere(Sphere<'b>),
    Quad(Quad<'b>),
    Triangle(Triangle<'b>),
}
impl_into_hitkind_non_recursive!(Sphere, Quad, Triangle);

#[repr(C)]
#[derive_builder('b)]
#[enum_dispatch(RecursiveHitable, IntoBoundingBox)]
pub enum HitKindRecursive<'b> {
    HitableList(HitableList<'b>),
    Translate(Translate<'b>),
    Rotate(Rotate<'b>),
}

impl_into_hitkind_recursive!(HitableList, Translate, Rotate);

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
