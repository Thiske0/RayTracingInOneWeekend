use crate::{RefBuilder, RefBuilderDevice, SliceBuilder, SliceBuilderDevice};
use cust::memory::DeviceBuffer;
use gpu_builder::{BuildResult, Builder, DeviceBufferList};
use std::{collections::HashMap, mem};

impl<'a, T: Builder<'a>> Builder<'a> for SliceBuilder<'a, T>
where
    <T as Builder<'a>>::Output: Sized,
{
    type Output = SliceBuilderDevice<T::Output>;

    unsafe fn build_device_inner(
        self,
        stream: &'a cust::prelude::Stream,
        cache: &mut gpu_builder::Cache<'a>,
    ) -> cust::error::CudaResult<gpu_builder::BuildResult<'a, Self>> {
        let entry = cache
            .get_mut::<HashMap<(u64, usize), *const T::Output>>()
            .or_else(|| {
                cache
                    .insert::<HashMap<(u64, usize), *const T::Output>>(HashMap::new())
                    .ok()?;
                cache.get_mut::<HashMap<(u64, usize), *const T::Output>>()
            })
            .expect("Just inserted, should exist");

        let (result, buffers): (*const T::Output, _) =
            if let Some(result) = entry.get(&(self.reference as *const () as u64, self.size)) {
                (*result, DeviceBufferList::new())
            } else {
                let mut result_buffers = DeviceBufferList::new();
                let build_result = self
                    .as_slice()
                    .into_iter()
                    .map(|item| {
                        let (result_device, result_host, buffers) = unsafe { std::ptr::read(item) }
                            .build_device_inner(stream, cache)?
                            .split();
                        mem::forget(result_host); // Prevent double free
                        result_buffers.combine(buffers);
                        Ok(result_device)
                    })
                    .try_collect::<Vec<T::Output>>()?;

                let device_buffer = DeviceBuffer::from_slice(build_result.as_slice())?;
                let result = device_buffer.as_device_ptr().as_ptr();
                result_buffers.add(device_buffer);
                cache
                    .get_mut::<HashMap<(u64, usize), *const T::Output>>()
                    .expect("Just inserted, should exist")
                    .insert((self.reference as *const () as u64, self.size), result);

                (result, result_buffers)
            };
        let result_host = SliceBuilder {
            reference: self.reference,
            size: self.size,
            _marker: core::marker::PhantomData,
            owned: self.owned,
        };
        mem::forget(self); // Prevent double free
        Ok(BuildResult::new(
            SliceBuilderDevice {
                reference: result,
                size: result_host.size,
                _marker: Default::default(),
            },
            result_host,
            stream,
            buffers,
        ))
    }

    fn copy_back(
        &mut self,
        _device_ref: &cust::memory::DeviceBox<Self::Output>,
    ) -> cust::error::CudaResult<()> {
        //No need to copy back a slice since it immutable
        Ok(())
    }
}

impl<'a, T: Builder<'a>> SliceBuilder<'a, T> {
    pub fn new(reference: &'a [T]) -> Self {
        SliceBuilder {
            reference: reference.as_ptr(),
            size: reference.len(),
            _marker: core::marker::PhantomData,
            owned: false,
        }
    }
}

impl<'a, T: Builder<'a>> SliceBuilder<'a, T> {
    pub fn new_owned(owned_reference: Vec<T>) -> Self {
        let size = owned_reference.len();
        let reference = owned_reference.as_ptr();
        core::mem::forget(owned_reference);

        SliceBuilder {
            reference,
            size,
            _marker: core::marker::PhantomData,
            owned: true,
        }
    }
}

impl<'a, T: Builder<'a>> Drop for SliceBuilder<'a, T> {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                let _ = Vec::from_raw_parts(self.reference as *mut T, self.size, self.size);
            }
        }
    }
}

impl<'a, T: Builder<'a>> Builder<'a> for RefBuilder<'a, T> {
    type Output = RefBuilderDevice<T::Output>;

    unsafe fn build_device_inner(
        self,
        stream: &'a cust::prelude::Stream,
        cache: &mut gpu_builder::Cache<'a>,
    ) -> cust::error::CudaResult<gpu_builder::BuildResult<'a, Self>> {
        let entry = cache
            .get_mut::<HashMap<u64, *const T::Output>>()
            .or_else(|| {
                cache
                    .insert::<HashMap<u64, *const T::Output>>(HashMap::new())
                    .ok()?;
                cache.get_mut::<HashMap<u64, *const T::Output>>()
            })
            .expect("Just inserted, should exist");

        let (result, buffers): (*const T::Output, _) = if let Some(result) =
            entry.get(&(self.reference as *const () as u64))
        {
            (*result, DeviceBufferList::new())
        } else {
            let mut result_buffers = DeviceBufferList::new();
            let (build_result, result_host, buffers) = unsafe { std::ptr::read(self.reference) }
                .build_device_inner(stream, cache)?
                .split();
            mem::forget(result_host); // Prevent double free
            result_buffers.combine(buffers);

            let device_buffer = DeviceBuffer::from_slice(&[build_result])?;
            let result = device_buffer.as_device_ptr().as_ptr();
            result_buffers.add(device_buffer);
            cache
                .get_mut::<HashMap<u64, *const T::Output>>()
                .expect("Just inserted, should exist")
                .insert(self.reference as *const () as u64, result);

            (result, result_buffers)
        };
        let result_host = RefBuilder {
            reference: self.reference,
            _marker: core::marker::PhantomData,
            owned: self.owned,
        };
        mem::forget(self); // Prevent double free
        Ok(BuildResult::new(
            RefBuilderDevice {
                reference: result,
                _marker: Default::default(),
            },
            result_host,
            stream,
            buffers,
        ))
    }

    fn copy_back(
        &mut self,
        _device_ref: &cust::memory::DeviceBox<Self::Output>,
    ) -> cust::error::CudaResult<()> {
        //No need to copy back a slice since it immutable
        Ok(())
    }
}

impl<'a, T: Builder<'a>> RefBuilder<'a, T> {
    pub fn new(reference: &'a T) -> Self {
        RefBuilder {
            reference,
            _marker: core::marker::PhantomData,
            owned: false,
        }
    }
}

impl<'a, T: Builder<'a>> RefBuilder<'a, T> {
    pub fn new_owned(owned_reference: T) -> Self {
        RefBuilder {
            reference: Box::leak(Box::new(owned_reference)),
            _marker: core::marker::PhantomData,
            owned: true,
        }
    }
}

impl<'a, T: Builder<'a>> Drop for RefBuilder<'a, T> {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                let _ = Box::from_raw(self.reference as *mut T);
            }
        }
    }
}
