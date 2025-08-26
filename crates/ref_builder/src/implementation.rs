use crate::{RefBuilder, RefBuilderDevice, SliceBuilder, SliceBuilderDevice};
use cust::memory::DeviceBuffer;
use gpu_builder::{BuildResult, Builder, DeviceBufferList};
use std::collections::HashMap;

impl<'a, T: Builder<'a>> Builder<'a> for SliceBuilder<'a, T>
where
    <T as Builder<'a>>::Output: Sized,
{
    type Output = SliceBuilderDevice<T::Output>;

    fn build_inner(self, cache: &mut gpu_builder::Cache<'a>) -> Self::Output {
        let entry = cache
            .get_mut::<HashMap<(u64, usize), Vec<T::Output>>>()
            .or_else(|| {
                cache
                    .insert::<HashMap<(u64, usize), Vec<T::Output>>>(HashMap::new())
                    .ok()?;
                cache.get_mut::<HashMap<(u64, usize), Vec<T::Output>>>()
            })
            .expect("Just inserted, should exist");
        let result: &Vec<T::Output> =
            if let Some(result) = entry.get(&(self.reference as u64, self.size)) {
                result
            } else {
                let build_result = self
                    .as_slice()
                    .into_iter()
                    .map(|item| unsafe { std::ptr::read(item) }.build_inner(cache))
                    .collect::<Vec<T::Output>>();

                entry.insert((self.reference as u64, self.size), build_result);
                entry
                    .get(&(self.reference as u64, self.size))
                    .expect("Just inserted, should exist")
            };
        SliceBuilderDevice {
            reference: result.as_ptr() as *const T::Output,
            size: self.size,
        }
    }

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

        let (result, buffers): (*const T::Output, _) = if let Some(result) =
            entry.get(&(self.reference as *const () as u64, self.size))
        {
            (*result, DeviceBufferList::new())
        } else {
            let mut result_buffers = DeviceBufferList::new();
            let build_result = self
                .as_slice()
                .into_iter()
                .map(|item| {
                    let (result_device, _result_host, buffers) = unsafe { std::ptr::read(item) }
                        .build_device_inner(stream, cache)?
                        .split();
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
        Ok(BuildResult::new(
            SliceBuilderDevice {
                reference: result,
                size: self.size,
            },
            SliceBuilder {
                reference: self.reference,
                size: self.size,
                _marker: core::marker::PhantomData,
            },
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
        }
    }
}

impl<'a, T: Builder<'a>> Builder<'a> for RefBuilder<'a, T> {
    type Output = RefBuilderDevice<T::Output>;

    fn build_inner(self, cache: &mut gpu_builder::Cache<'a>) -> Self::Output {
        let entry = cache
            .get_mut::<HashMap<u64, Vec<T::Output>>>()
            .or_else(|| {
                cache
                    .insert::<HashMap<u64, Vec<T::Output>>>(HashMap::new())
                    .ok()?;
                cache.get_mut::<HashMap<u64, Vec<T::Output>>>()
            })
            .expect("Just inserted, should exist");
        let result: &Vec<T::Output> = if let Some(result) = entry.get(&(self.reference as u64)) {
            result
        } else {
            let build_result = vec![unsafe { std::ptr::read(self.reference) }.build_inner(cache)];

            entry.insert(self.reference as u64, build_result);
            entry
                .get(&(self.reference as u64))
                .expect("Just inserted, should exist")
        };
        RefBuilderDevice {
            reference: result.as_ptr() as *const T::Output,
        }
    }

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
            let (build_result, _result_host, buffers) = unsafe { std::ptr::read(self.reference) }
                .build_device_inner(stream, cache)?
                .split();
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
        Ok(BuildResult::new(
            RefBuilderDevice { reference: result },
            RefBuilder {
                reference: self.reference,
                _marker: core::marker::PhantomData,
            },
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
        }
    }
}
