use core::ops::Range;

use crate::boundingbox::IntoBoundingBox;
use crate::{
    boundingbox::BoundingBox,
    hitable::{HitKind, HitRecord, Hitable},
    ray::Ray,
    vec3::Real,
};

#[cfg(not(target_os = "cuda"))]
pub struct HitableListBuilder<'a> {
    hitables: Vec<HitKind<'a>>,
    bounding_box: BoundingBox,
    builders: Vec<BuilderKind<'a>>,
}

#[cfg(not(target_os = "cuda"))]
use cust::{
    error::CudaResult,
    memory::{DeviceBox, DeviceBuffer, DeviceCopy},
    stream::Stream,
};

#[cfg(not(target_os = "cuda"))]
use enum_dispatch::enum_dispatch;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy))]
pub struct HitableList<'a> {
    hitables: &'a [HitKind<'a>],
    bounding_box: BoundingBox,
}
#[cfg(not(target_os = "cuda"))]
unsafe impl<'a> DeviceCopy for HitableList<'a> {}

impl IntoBoundingBox for HitableList<'_> {
    fn boundingbox(&self) -> BoundingBox {
        BoundingBox::empty().merge(&self.bounding_box)
    }
}

#[cfg(not(target_os = "cuda"))]
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

impl Hitable for HitableList<'_> {
    fn hit<'a>(&'a self, ray: &Ray, interval: &Range<Real>) -> Option<HitRecord<'a>> {
        if !self.bounding_box.hit(ray, interval) {
            return None;
        }

        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_interval = interval.clone();

        for hitable in self.hitables {
            if let Some(hit_record) = hitable.hit(ray, &closest_interval) {
                closest_interval = closest_interval.start..hit_record.t;
                closest_hit = Some(hit_record);
            }
        }

        closest_hit
    }
}

#[cfg(not(target_os = "cuda"))]
#[enum_dispatch]
pub trait Builder<'a> {
    fn build(&'a mut self) -> HitKind<'a>;
    unsafe fn build_device(
        &'a mut self,
        stream: &Stream,
    ) -> CudaResult<(DeviceBox<HitKind<'a>>, DeviceBufferList<'a>)>;
}

#[cfg(not(target_os = "cuda"))]
pub struct DeviceBufferList<'a> {
    buffers: Vec<DeviceBuffer<HitKind<'a>>>,
}

#[cfg(not(target_os = "cuda"))]
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

#[cfg(not(target_os = "cuda"))]
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

        HitableList {
            hitables: self.hitables.as_slice(),
            bounding_box: self.bounding_box,
        }
        .into()
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
        let hitable_list = HitableList {
            hitables: unsafe {
                std::slice::from_raw_parts(
                    device_buffer.as_device_ptr().as_ptr(),
                    device_buffer.len(),
                )
            },
            bounding_box: self.bounding_box,
        };
        device_buffer_list.add(device_buffer);

        Ok((
            unsafe { DeviceBox::new_async(&hitable_list.into(), stream)? },
            device_buffer_list,
        ))
    }
}

#[cfg(not(target_os = "cuda"))]
#[enum_dispatch(Builder)]
pub enum BuilderKind<'a> {
    HitableListBuilder(HitableListBuilder<'a>),
}
