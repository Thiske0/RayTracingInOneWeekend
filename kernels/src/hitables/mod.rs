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
    fn pdf_value(&self, origin: &Point3, direction: &Vec3, rng: &mut Random) -> Real;
    fn random(&self, origin: &Point3, rng: &mut Random) -> Vec3;
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
}

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
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

trait Recursive<'a> {
    fn recurse(
        &mut self,
        hitkind: &'a HitKindRecursive<'a>,
        count: usize,
    ) -> Option<(&'a HitKind<'a>, usize)>;
    fn handle_non_recursive(&mut self, hitkind: &'a HitKindNonRecursive<'a>);
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
        hitkind: &'a HitKindRecursive<'a>,
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

    fn handle_non_recursive(&mut self, hitkind: &'a HitKindNonRecursive<'a>) {
        if let Some(rec) = hitkind.hit(&self.ray, &self.range, &mut self.rng) {
            if let Some(current_rec) = &self.hit_record {
                if rec.t < current_rec.t {
                    self.range = self.range.start..rec.t;
                    self.hit_record = Some(rec);
                }
            } else {
                self.range = self.range.start..rec.t;
                self.hit_record = Some(rec);
            }
        }
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
        hitkind: &'a HitKindRecursive<'a>,
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

    fn handle_non_recursive(&mut self, hitkind: &'a HitKindNonRecursive<'a>) {
        let value = hitkind.pdf_value(&self.origin, &self.direction, self.rng);
        self.current_value += value;
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
        hitkind: &'a HitKindRecursive<'a>,
        count: usize,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        hitkind.random_recursive(count, &mut self.origin, &mut self.current_value, self.rng)
    }

    fn handle_non_recursive(&mut self, hitkind: &'a HitKindNonRecursive<'a>) {
        let value = hitkind.random(&self.origin, self.rng);
        self.current_value += value;
    }
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

    pub fn pdf_value(&self, origin: &Point3, direction: &Vec3, rng: &mut Random) -> Real {
        match self {
            HitKind::HitKindNonRecursive(h) => h.pdf_value(origin, direction, rng),
            HitKind::HitKindRecursive(h) => Self::pdf_value_recursive(h, origin, direction, rng),
        }
    }

    pub fn random(&self, origin: &Point3, rng: &mut Random) -> Vec3 {
        match self {
            HitKind::HitKindNonRecursive(h) => h.random(origin, rng),
            HitKind::HitKindRecursive(h) => Self::random_recursive(h, origin, rng),
        }
    }

    fn hit_recursive<'a>(
        hitkind: &'a HitKindRecursive<'a>,
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

        Self::use_stack(hitkind, &mut recurse_impl);
        return recurse_impl.hit_record;
    }

    fn pdf_value_recursive<'a>(
        hitkind: &'a HitKindRecursive<'a>,
        origin: &Point3,
        direction: &Vec3,
        rng: &mut Random,
    ) -> Real {
        let mut recurse_impl = PDFValueRecursive {
            origin: origin.clone(),
            direction: direction.clone(),
            current_value: 0.0,
            rng: rng,
        };

        Self::use_stack(hitkind, &mut recurse_impl);
        return recurse_impl.current_value;
    }

    fn random_recursive<'a>(
        hitkind: &'a HitKindRecursive<'a>,
        origin: &Point3,
        rng: &mut Random,
    ) -> Vec3 {
        let mut recurse_impl = PDFRandomRecursive {
            origin: origin.clone(),
            current_value: Vec3::zero(),
            rng: rng,
        };

        Self::use_stack(hitkind, &mut recurse_impl);
        return recurse_impl.current_value;
    }

    fn use_stack<'a, RecursiveImpl: Recursive<'a>>(
        initial: &'a HitKindRecursive<'a>,
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
                match inner_hitkind {
                    HitKind::HitKindNonRecursive(inner_hitkind) => {
                        recurse_impl.handle_non_recursive(inner_hitkind);
                    }
                    HitKind::HitKindRecursive(inner_hitkind) => {
                        stack_ptr += 1;
                        stack[stack_ptr - 1] = StackEntry::new(inner_hitkind);
                    }
                }
            } else {
                stack_ptr -= 1;
                if stack_ptr == 0 {
                    return;
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
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[enum_dispatch(Hitable, IntoBoundingBox)]
#[derive_builder('b)]
pub enum HitKindNonRecursive<'b> {
    Sphere(Sphere<'b>),
    Quad(Quad<'b>),
    Triangle(Triangle<'b>),
}
impl_into_hitkind_non_recursive!(Sphere, Quad, Triangle);

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
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

    fn pdf_value_recursive<'a>(
        &'a self,
        count: usize,
        origin: &mut Point3,
        direction: &mut Vec3,
        current_value: &mut Real,
        rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        match self {
            HitKindRecursive::HitableList(h) => {
                h.pdf_value_recursive(count, origin, direction, current_value, rng)
            }
            HitKindRecursive::Translate(h) => {
                h.pdf_value_recursive(count, origin, direction, current_value, rng)
            }
            HitKindRecursive::Rotate(h) => {
                h.pdf_value_recursive(count, origin, direction, current_value, rng)
            }
            HitKindRecursive::ConstantMedium(h) => {
                h.pdf_value_recursive(count, origin, direction, current_value, rng)
            }
        }
    }

    fn random_recursive<'a>(
        &'a self,
        count: usize,
        origin: &mut Point3,
        current_value: &mut Vec3,
        rng: &mut Random,
    ) -> Option<(&'a HitKind<'a>, usize)> {
        match self {
            HitKindRecursive::HitableList(h) => {
                h.random_recursive(count, origin, current_value, rng)
            }
            HitKindRecursive::Translate(h) => h.random_recursive(count, origin, current_value, rng),
            HitKindRecursive::Rotate(h) => h.random_recursive(count, origin, current_value, rng),
            HitKindRecursive::ConstantMedium(h) => {
                h.random_recursive(count, origin, current_value, rng)
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
