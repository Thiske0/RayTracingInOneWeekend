use core::ops::Range;

use crate::{
    hitable::{HitKind, HitRecord, Hitable},
    ray::Ray,
    vec3::Real,
};

#[cfg(not(target_os = "cuda"))]
pub struct HitableListBuilder<'a> {
    hitables: Vec<HitKind<'a>>,
}

#[cfg(not(target_os = "cuda"))]
use cust::{
    error::CudaResult,
    memory::{DeviceBox, DeviceBuffer, DeviceCopy},
    stream::Stream,
};

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy))]
pub struct HitableList<'a> {
    hitables: &'a [HitKind<'a>],
}
#[cfg(not(target_os = "cuda"))]
unsafe impl<'a> DeviceCopy for HitableList<'a> {}

#[cfg(not(target_os = "cuda"))]
impl<'a> HitableListBuilder<'a> {
    pub fn new() -> Self {
        HitableListBuilder {
            hitables: Vec::new(),
        }
    }

    pub fn add(&mut self, hitable: HitKind<'a>) {
        self.hitables.push(hitable);
    }

    /// Builds the HitableList from the added hitables.
    /// This is for CPU usage only.
    /// For GPU usage, use `HitableList::build_device`.
    /// If you try using this on the GPU, it will invoke UB.
    pub fn build(&'a self) -> HitKind<'a> {
        HitableList {
            hitables: self.hitables.as_slice(),
        }
        .into()
    }

    pub unsafe fn build_device(
        &self,
        stream: &Stream,
    ) -> CudaResult<(DeviceBox<HitKind<'a>>, DeviceBuffer<HitKind<'a>>)> {
        let device_buffer =
            unsafe { DeviceBuffer::from_slice_async(self.hitables.as_slice(), stream)? };
        let hitable_list = HitableList {
            hitables: unsafe {
                std::slice::from_raw_parts(
                    device_buffer.as_device_ptr().as_ptr(),
                    device_buffer.len(),
                )
            },
        };
        Ok((
            unsafe { DeviceBox::new_async(&hitable_list.into(), stream)? },
            device_buffer,
        ))
    }
}

impl Hitable for HitableList<'_> {
    fn hit<'a>(&'a self, ray: &Ray, interval: &Range<Real>) -> Option<HitRecord<'a>> {
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
