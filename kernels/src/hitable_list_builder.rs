use std::mem;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitable::HitKind,
    hitable_list::HitableList,
};
use enum_dispatch::enum_dispatch;

impl<'a> IntoBoundingBox for HitableListBuilder<'a> {
    fn boundingbox(&self) -> BoundingBox {
        self.hitables
            .iter()
            .fold(BoundingBox::empty(), |acc, hitable| {
                acc.merge(&hitable.boundingbox())
            })
    }
}

#[enum_dispatch(IntoBoundingBox)]
enum HitableListBuilderKind<'a> {
    Leaf(HitKind<'a>),
    Node(HitableListBuilder<'a>),
}

#[cfg(not(target_os = "cuda"))]
pub struct HitableListBuilder<'a> {
    hitables: Vec<HitableListBuilderKind<'a>>,
    build_result: Vec<HitKind<'a>>,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> HitableListBuilder<'a> {
    pub fn new() -> Self {
        HitableListBuilder {
            hitables: Vec::new(),
            build_result: Vec::new(),
        }
    }

    fn split(self) -> (HitableListBuilder<'a>, Option<HitableListBuilder<'a>>) {
        let bounding_box = self.boundingbox();

        let axis = bounding_box.longest_axis();
        let average_along_axis = (&bounding_box.center())[&axis];

        let (mut left_hitables, mut right_hitables): (
            Vec<HitableListBuilderKind>,
            Vec<HitableListBuilderKind>,
        ) = self.hitables.into_iter().partition(|h| {
            let center = h.boundingbox().center();
            (&center)[&axis] < average_along_axis
        });

        if left_hitables.len() < right_hitables.len() {
            mem::swap(&mut left_hitables, &mut right_hitables);
        }

        if !right_hitables.is_empty() {
            (
                HitableListBuilder {
                    hitables: left_hitables,
                    build_result: Vec::new(),
                },
                None,
            )
        } else {
            (
                HitableListBuilder {
                    hitables: left_hitables,
                    build_result: Vec::new(),
                },
                Some(HitableListBuilder {
                    hitables: right_hitables,
                    build_result: Vec::new(),
                }),
            )
        }
    }

    pub fn subdivide(self, divisions: &[usize]) -> HitableListBuilder<'a> {
        if divisions.is_empty() {
            return self;
        }

        let times = divisions[0];
        let mut divided = vec![self];
        for _ in 0..times {
            divided = divided
                .into_iter()
                .flat_map(|builder| {
                    let (left, right) = builder.split();
                    if let Some(right) = right {
                        vec![left, right]
                    } else {
                        vec![left]
                    }
                })
                .collect();
        }

        let builders = divided
            .into_iter()
            .map(|builder| HitableListBuilderKind::Node(builder.subdivide(&divisions[1..])))
            .collect::<Vec<_>>();

        HitableListBuilder {
            hitables: builders,
            build_result: Vec::new(),
        }
    }

    pub fn add(&mut self, hitable: HitKind<'a>) {
        self.hitables.push(HitableListBuilderKind::Leaf(hitable));
    }

    pub fn add_unrolled(&mut self, other: HitableListBuilder<'a>) {
        self.hitables.extend(other.hitables);
    }
}

impl<'a> From<&'a mut HitableListBuilder<'a>> for HitableList<'a> {
    fn from(builder: &'a mut HitableListBuilder<'a>) -> Self {
        builder.build_result = builder
            .hitables
            .iter_mut()
            .map(|h| match h {
                HitableListBuilderKind::Leaf(h) => unsafe { std::ptr::read(h) },
                HitableListBuilderKind::Node(b) => HitKind::from(b),
            })
            .collect();
        HitableList::new(builder.build_result.as_slice())
    }
}

impl<'a> From<&'a mut HitableListBuilder<'a>> for HitKind<'a> {
    fn from(builder: &'a mut HitableListBuilder<'a>) -> Self {
        Into::<HitableList<'a>>::into(builder).into()
    }
}
