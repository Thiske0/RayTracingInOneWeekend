use std::mem;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitables::HitKind,
    hitables::hitable_list::HitableList,
    vec3::Real,
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

pub enum SlitMethod {
    Middle,
    Sah,
    SahBy4,
}

#[cfg(not(target_os = "cuda"))]
impl<'a> HitableListBuilder<'a> {
    pub fn new() -> Self {
        HitableListBuilder {
            hitables: Vec::new(),
        }
    }

    fn split_middle(self) -> (HitableListBuilder<'a>, Option<HitableListBuilder<'a>>) {
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

    fn split_sah<const BINS: usize>(
        self,
    ) -> (HitableListBuilder<'a>, Option<HitableListBuilder<'a>>) {
        if self.hitables.len() <= 8 {
            return (self, None);
        }

        let bbox = self.boundingbox();
        let axis = bbox.longest_axis();
        let axis_min = bbox.min.at_axis(&axis);
        let axis_max = bbox.max.at_axis(&axis);
        let bin_width = (axis_max - axis_min) / (BINS as Real);

        // Initialize bins
        let mut bins = vec![(BoundingBox::empty(), 0); BINS];

        // Assign objects to bins and compute bin bounds
        for hitable in &self.hitables {
            let center = hitable.boundingbox().center().at_axis(&axis);
            let bin_index = ((center - axis_min) / bin_width).floor() as usize;
            let bin_index = bin_index.min(BINS - 1);

            bins[bin_index].0 = bins[bin_index].0.merge(&hitable.boundingbox());
            bins[bin_index].1 += 1;
        }

        // Find best split using SAH
        let mut best_cost = Real::INFINITY;
        let mut best_split = 0;

        // Precompute prefix sums
        let mut left_bbox = BoundingBox::empty();
        let mut left_count = 0;

        for i in 0..BINS - 1 {
            if bins[i].1 > 0 {
                left_bbox = left_bbox.merge(&bins[i].0);
                left_count += bins[i].1;
            }

            let right_bbox = (i + 1..BINS)
                .filter_map(|j| {
                    if bins[j].1 > 0 {
                        Some(&bins[j].0)
                    } else {
                        None
                    }
                })
                .fold(BoundingBox::empty(), |acc, b| acc.merge(b));
            let right_count: usize = (i + 1..BINS).map(|j| bins[j].1).sum();

            if left_count > 0 && right_count > 0 {
                let cost = (left_count as Real) * left_bbox.surface_area()
                    + (right_count as Real) * right_bbox.surface_area();

                if cost < best_cost {
                    best_cost = cost;
                    best_split = i;
                }
            }
        }

        // Partition based on best split
        let split_pos = axis_min + (best_split + 1) as f32 * bin_width;
        let (left, right): (Vec<_>, Vec<_>) = self
            .hitables
            .into_iter()
            .partition(|h| h.boundingbox().center().at_axis(&axis) < split_pos);

        if right.is_empty() {
            HitableListBuilder { hitables: left }.split_middle()
        } else {
            (
                HitableListBuilder { hitables: left },
                Some(HitableListBuilder { hitables: right }),
            )
        }
    }

    pub fn subdivide_by4(
        self,
        times: usize,
        method: SlitMethod,
        sort_nodes: bool,
    ) -> HitableListBuilder<'a> {
        let divisions = vec![2; times];
        self.subdivide(divisions.as_slice(), &method, sort_nodes)
    }

    pub fn subdivide(
        self,
        divisions: &[usize],
        method: &SlitMethod,
        sort_nodes: bool,
    ) -> HitableListBuilder<'a> {
        if divisions.is_empty() {
            return self;
        }

        let times = divisions[0];
        let mut divided = vec![self];
        for _ in 0..times {
            divided = divided
                .into_iter()
                .flat_map(|builder| {
                    let (left, right) = match method {
                        SlitMethod::Middle => builder.split_middle(),
                        SlitMethod::Sah => builder.split_sah::<32>(),
                        SlitMethod::SahBy4 => unimplemented!(),
                    };
                    if let Some(right) = right {
                        vec![right, left]
                    } else {
                        vec![left]
                    }
                })
                .collect();
        }

        let mut builders = divided
            .into_iter()
            .map(|builder| builder.subdivide(&divisions[1..], method, sort_nodes))
            .collect::<Vec<_>>();

        if sort_nodes {
            let bounding_box = builders
                .iter()
                .fold(BoundingBox::empty(), |acc, b| acc.merge(&b.boundingbox()));
            let axis = bounding_box.longest_axis();
            builders.sort_by(|a, b| {
                a.boundingbox()
                    .center()
                    .at_axis(&axis)
                    .partial_cmp(&b.boundingbox().center().at_axis(&axis))
                    .unwrap()
            });
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

impl<'a> From<HitableListBuilder<'a>> for HitKind<'a> {
    fn from(builder: HitableListBuilder<'a>) -> Self {
        HitableList::new(builder.hitables).into()
    }
}
