use std::mem;

use crate::{
    boundingbox::{BoundingBox, IntoBoundingBox},
    hitable::HitKind,
    hitable_list::HitableList,
};

use cust::{
    error::CudaResult,
    memory::{DeviceBox, DeviceBuffer},
    stream::Stream,
};

use enum_dispatch::enum_dispatch;

pub struct HitableListBuilder<'a> {
    hitables: Vec<HitKind<'a>>,
    bounding_box: BoundingBox,
    builders: Vec<BuilderKind<'a>>,
}

impl<'a> HitableListBuilder<'a> {
    pub fn new() -> Self {
        HitableListBuilder {
            hitables: Vec::new(),
            bounding_box: BoundingBox::empty(),
            builders: Vec::new(),
        }
    }

    fn from_parts(hitables: Vec<HitKind<'a>>, builders: Vec<BuilderKind<'a>>) -> Self {
        let bounding_box = hitables
            .iter()
            .fold(BoundingBox::empty(), |acc, h| acc.merge(&h.boundingbox()));
        HitableListBuilder {
            hitables,
            bounding_box,
            builders,
        }
    }

    pub fn add(&mut self, hitable: HitKind<'a>) {
        self.bounding_box = self.bounding_box.merge(&hitable.boundingbox());
        self.hitables.push(hitable);
    }

    pub fn add_builder(&mut self, builder: BuilderKind<'a>) {
        self.builders.push(builder);
    }

    pub fn add_unrolled(&mut self, builder: HitableListBuilder<'a>) {
        self.bounding_box = self.bounding_box.merge(&builder.boundingbox());
        self.builders.extend(builder.builders);
        self.hitables.extend(builder.hitables);
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

        let (mut left_builders, mut right_builders): (Vec<BuilderKind<'a>>, Vec<BuilderKind<'a>>) =
            self.builders.into_iter().partition(|h| {
                let center = h.boundingbox().center();
                (&center)[&axis] < average_along_axis
            });

        if left_hitables.len() + left_builders.len() < right_hitables.len() + right_builders.len() {
            mem::swap(&mut left_hitables, &mut right_hitables);
            mem::swap(&mut left_builders, &mut right_builders);
        }

        if !right_hitables.is_empty() && !right_builders.is_empty() {
            (
                HitableListBuilder::from_parts(left_hitables, left_builders),
                None,
            )
        } else {
            (
                HitableListBuilder::from_parts(left_hitables, left_builders),
                Some(HitableListBuilder::from_parts(
                    right_hitables,
                    right_builders,
                )),
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

        let divided: Vec<BuilderKind<'a>> = divided
            .into_iter()
            .map(|builder| builder.subdivide(&divisions[1..]).into())
            .collect();
        HitableListBuilder::from_parts(Vec::new(), divided)
    }
}

#[enum_dispatch]
pub trait Builder<'a> {
    fn build(&'a mut self) -> HitKind<'a>;
    unsafe fn build_device(
        &'a mut self,
        stream: &Stream,
    ) -> CudaResult<(DeviceBox<HitKind<'a>>, DeviceBufferList<'a>)>;
}

pub struct DeviceBufferList<'a> {
    buffers: Vec<DeviceBuffer<HitKind<'a>>>,
}

impl<'a> DeviceBufferList<'a> {
    pub fn new() -> Self {
        DeviceBufferList {
            buffers: Vec::new(),
        }
    }

    pub fn add(&mut self, buffer: DeviceBuffer<HitKind<'a>>) {
        self.buffers.push(buffer);
    }

    pub fn combine(&mut self, other: DeviceBufferList<'a>) {
        self.buffers.extend(other.buffers);
    }

    pub fn drop_async(self, stream: &Stream) -> CudaResult<()> {
        for buffer in self.buffers {
            buffer.drop_async(stream)?;
        }
        Ok(())
    }
}

impl<'a> Builder<'a> for HitableListBuilder<'a> {
    /// Builds the HitableList from the added hitables.
    /// This is for CPU usage only.
    /// For GPU usage, use `HitableList::build_device`.
    /// If you try using this on the GPU, it will invoke UB.
    fn build(&'a mut self) -> HitKind<'a> {
        for builder in &mut self.builders {
            let hitable = builder.build();
            self.hitables.push(hitable);
            self.bounding_box = self.bounding_box.merge(&hitable.boundingbox());
        }

        HitableList::new(self.hitables.as_slice(), self.bounding_box).into()
    }

    unsafe fn build_device(
        &'a mut self,
        stream: &Stream,
    ) -> CudaResult<(DeviceBox<HitKind<'a>>, DeviceBufferList<'a>)> {
        let mut device_buffer_list = DeviceBufferList::new();

        for builder in &mut self.builders {
            let (hitable, device_buffer) = unsafe { builder.build_device(stream)? };
            let hitable = hitable.as_host_value()?;
            self.hitables.push(hitable);
            self.bounding_box = self.bounding_box.merge(&hitable.boundingbox());
            device_buffer_list.combine(device_buffer);
        }

        let device_buffer =
            unsafe { DeviceBuffer::from_slice_async(self.hitables.as_slice(), stream)? };
        let hitable_list = HitableList::new(
            unsafe {
                if device_buffer.len() > 0 {
                    std::slice::from_raw_parts(
                        device_buffer.as_device_ptr().as_ptr(),
                        device_buffer.len(),
                    )
                } else {
                    &[]
                }
            },
            self.bounding_box,
        );
        device_buffer_list.add(device_buffer);

        Ok((
            unsafe { DeviceBox::new_async(&hitable_list.into(), stream)? },
            device_buffer_list,
        ))
    }
}

#[enum_dispatch(Builder, IntoBoundingBox)]
pub enum BuilderKind<'a> {
    HitableListBuilder(HitableListBuilder<'a>),
}

impl IntoBoundingBox for HitableListBuilder<'_> {
    fn boundingbox(&self) -> BoundingBox {
        self.builders
            .iter()
            .fold(self.bounding_box, |acc, builder| {
                acc.merge(&builder.boundingbox())
            })
    }
}
