use std::mem;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::HitKind,
    hitables::hitable_list::HitableList,
};

impl<'a> IntoBoundingBox for HitableListBuilder<'a> {
    fn boundingbox(&self) -> BoundingBox {
        self.hitables
            .iter()
            .fold(BoundingBox::empty(), |acc, hitable| {
                acc.merge(&hitable.boundingbox())
            })
    }
}

#[cfg(not(target_os = "cuda"))]
pub struct HitableListBuilder<'a> {
    hitables: Vec<HitKind<'a>>,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> HitableListBuilder<'a> {
    pub fn new() -> Self {
        HitableListBuilder {
            hitables: Vec::new(),
        }
    }

    fn split(self) -> (HitableListBuilder<'a>, Option<HitableListBuilder<'a>>) {
        let bounding_box = self.boundingbox();

        let axis = bounding_box.longest_axis();
        let average_along_axis = (&bounding_box.center())[&axis];

        let (mut left_hitables, mut right_hitables): (Vec<HitKind<'a>>, Vec<HitKind<'a>>) =
            self.hitables.into_iter().partition(|h| {
                let center = h.boundingbox().center();
                (&center)[&axis] < average_along_axis
            });

        if left_hitables.len() < right_hitables.len() {
            mem::swap(&mut left_hitables, &mut right_hitables);
        }

        if right_hitables.is_empty() {
            (
                HitableListBuilder {
                    hitables: left_hitables,
                },
                None,
            )
        } else {
            (
                HitableListBuilder {
                    hitables: left_hitables,
                },
                Some(HitableListBuilder {
                    hitables: right_hitables,
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
            .map(|builder| builder.subdivide(&divisions[1..]))
            .collect::<Vec<_>>();

        if builders.len() == 1 {
            let first = builders.into_iter().next().unwrap();
            return first;
        }

        HitableListBuilder {
            hitables: builders.into_iter().map(|b| b.into()).collect(),
        }
    }

    pub fn add(&mut self, hitable: HitKind<'a>) {
        self.hitables.push(hitable);
    }

    pub fn add_unrolled(&mut self, other: HitableListBuilder<'a>) {
        self.hitables.extend(other.hitables);
    }
}

impl<'a> From<&'a HitableListBuilder<'a>> for HitKind<'a> {
    fn from(builder: &'a HitableListBuilder<'a>) -> Self {
        HitableList::new(builder.hitables.as_slice()).into()
    }
}

impl<'a> From<HitableListBuilder<'a>> for HitKind<'a> {
    fn from(builder: HitableListBuilder<'a>) -> Self {
        HitableList::new_owned(builder.hitables).into()
    }
}
