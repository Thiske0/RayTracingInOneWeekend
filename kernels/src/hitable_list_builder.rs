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

    pub fn add(&mut self, hitable: HitKind<'a>) {
        self.bounding_box = self.bounding_box.merge(&hitable.boundingbox());
        self.hitables.push(hitable);
    }

    pub fn add_builder(&mut self, builder: BuilderKind<'a>) {
        self.builders.push(builder);
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
                std::slice::from_raw_parts(
                    device_buffer.as_device_ptr().as_ptr(),
                    device_buffer.len(),
                )
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

#[enum_dispatch(Builder)]
pub enum BuilderKind<'a> {
    HitableListBuilder(HitableListBuilder<'a>),
}
