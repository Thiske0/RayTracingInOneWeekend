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

#[derive(PartialEq)]
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

    fn split_middle(self) -> Vec<HitableListBuilder<'a>> {
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
            vec![HitableListBuilder {
                hitables: left_hitables,
            }]
        } else {
            vec![
                HitableListBuilder {
                    hitables: left_hitables,
                },
                HitableListBuilder {
                    hitables: right_hitables,
                },
            ]
        }
    }

    fn split_sah<const BINS: usize>(self) -> Vec<HitableListBuilder<'a>> {
        if self.hitables.len() <= 8 {
            return vec![self];
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

        if left.is_empty() {
            HitableListBuilder { hitables: right }.split_middle()
        } else if right.is_empty() {
            HitableListBuilder { hitables: left }.split_middle()
        } else {
            vec![
                HitableListBuilder { hitables: left },
                HitableListBuilder { hitables: right },
            ]
        }
    }

    fn split_sah_by4<const BINS: usize>(self) -> Vec<HitableListBuilder<'a>> {
        if self.hitables.len() <= 8 {
            return vec![self];
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
        let mut best_split = (0, 0, 0);

        // Precompute prefix sums
        let mut left_bbox = BoundingBox::empty();
        let mut left_count = 0;

        for i in 0..BINS - 3 {
            if bins[i].1 > 0 {
                left_bbox = left_bbox.merge(&bins[i].0);
                left_count += bins[i].1;
            }
            if left_count == 0 {
                continue;
            }
            let mut left_middle_bbox = BoundingBox::empty();
            let mut left_middle_count = 0;
            for j in (i + 1)..BINS - 2 {
                if bins[j].1 > 0 {
                    left_middle_bbox = left_middle_bbox.merge(&bins[j].0);
                    left_middle_count += bins[j].1;
                }
                if left_middle_count == 0 {
                    continue;
                }
                let mut right_middle_bbox = BoundingBox::empty();
                let mut right_middle_count = 0;
                for k in (j + 1)..BINS - 1 {
                    if bins[k].1 > 0 {
                        right_middle_bbox = right_middle_bbox.merge(&bins[k].0);
                        right_middle_count += bins[k].1;
                    }
                    if right_middle_count == 0 {
                        continue;
                    }

                    let right_bbox = (k + 1..BINS)
                        .filter_map(|l| {
                            if bins[l].1 > 0 {
                                Some(&bins[l].0)
                            } else {
                                None
                            }
                        })
                        .fold(BoundingBox::empty(), |acc, b| acc.merge(b));
                    let right_count: usize = (k + 1..BINS).map(|l| bins[l].1).sum();

                    if right_count == 0 {
                        continue;
                    }

                    let cost = (left_count as Real) * left_bbox.surface_area()
                        + (left_middle_count as Real) * left_middle_bbox.surface_area()
                        + (right_middle_count as Real) * right_middle_bbox.surface_area()
                        + (right_count as Real) * right_bbox.surface_area();
                    if cost < best_cost {
                        best_cost = cost;
                        best_split = (i, j, k);
                    }
                }
            }
        }

        // Partition based on best split
        let split_pos = axis_min + (best_split.1 + 1) as f32 * bin_width;
        let (left, right): (Vec<_>, Vec<_>) = self
            .hitables
            .into_iter()
            .partition(|h| h.boundingbox().center().at_axis(&axis) < split_pos);
        let split_pos = axis_min + (best_split.0 + 1) as f32 * bin_width;
        let (left_left, left_middle): (Vec<_>, Vec<_>) = left
            .into_iter()
            .partition(|h| h.boundingbox().center().at_axis(&axis) < split_pos);
        let split_pos = axis_min + (best_split.2 + 1) as f32 * bin_width;
        let (right_middle, right_right): (Vec<_>, Vec<_>) = right
            .into_iter()
            .partition(|h| h.boundingbox().center().at_axis(&axis) < split_pos);

        let mut result = Vec::new();
        if !left_left.is_empty() {
            result.push(HitableListBuilder {
                hitables: left_left,
            });
        }
        if !left_middle.is_empty() {
            result.push(HitableListBuilder {
                hitables: left_middle,
            });
        }
        if !right_middle.is_empty() {
            result.push(HitableListBuilder {
                hitables: right_middle,
            });
        }
        if !right_right.is_empty() {
            result.push(HitableListBuilder {
                hitables: right_right,
            });
        }
        if result.len() < 2 {
            let recombined = result
                .into_iter()
                .flat_map(|b| b.hitables)
                .collect::<Vec<_>>();
            return HitableListBuilder {
                hitables: recombined,
            }
            .split_sah::<BINS>();
        }
        result
    }

    pub fn subdivide_by4(
        self,
        times: usize,
    ) -> HitableListBuilder<'a> {
        let divisions = vec![2; times];
        self.subdivide(divisions.as_slice())
    }

    pub fn subdivide(
        self,
        divisions: &[usize],
    ) -> HitableListBuilder<'a> {
        if divisions.is_empty() {
            return self;
        }

        let  times = (divisions[0] + 1) / 2;
        let mut divided = vec![self];
        for _ in 0..times {
            divided = divided
                .into_iter()
                .flat_map(|builder| 
                    builder.split_sah_by4::<32>()
                )
                .collect();
        }

        let mut builders = divided
            .into_iter()
            .map(|builder| builder.subdivide(&divisions[1..]))
            .collect::<Vec<_>>();

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
