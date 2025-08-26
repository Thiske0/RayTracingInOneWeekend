#[cfg(not(target_os = "cuda"))]
pub use cache::*;
pub use gpu_builder_macros::*;
#[cfg(not(target_os = "cuda"))]
pub use host_impls::*;

#[cfg(not(target_os = "cuda"))]
pub mod cache;

#[cfg(target_os = "cuda")]
pub trait Builder<'a> {
    type Output: BuildResultType;
}

#[cfg(target_os = "cuda")]
pub trait BuildResultType: 'static {}
#[cfg(target_os = "cuda")]
mod gpu_defaults;

#[cfg(not(target_os = "cuda"))]
mod host_impls {
    use std::{any::Any, cell::RefCell};

    use cust::{
        error::{CudaError, CudaResult},
        memory::{CopyDestination, DeviceBox, DeviceBuffer, DeviceCopy, DevicePointer},
        stream::Stream,
    };

    use crate::cache::Cache;
    pub trait BuildResultType: DeviceCopy + 'static {}

    pub trait Builder<'a>: Sized + 'a {
        type Output: BuildResultType;
        fn build_inner(self, cache: &mut Cache<'a>) -> Self::Output;
        unsafe fn build_device_inner(
            self,
            stream: &'a Stream,
            cache: &mut Cache<'a>,
        ) -> CudaResult<BuildResult<'a, Self>>;
        fn copy_back(&mut self, device_box: &DeviceBox<Self::Output>) -> CudaResult<()>;
    }

    pub trait NiceBuilder<'a>: Builder<'a> {
        fn build(self) -> Self::Output;
        unsafe fn build_device(self, stream: &'a Stream) -> CudaResult<BuildResult<'a, Self>>;
    }

    impl<'a, T: Builder<'a>> NiceBuilder<'a> for T {
        fn build(self) -> Self::Output {
            let mut cache = Cache::new();
            let output = self.build_inner(&mut cache);
            output
        }
        unsafe fn build_device(self, stream: &'a Stream) -> CudaResult<BuildResult<'a, Self>> {
            let mut cache = Cache::new();
            self.build_device_inner(stream, &mut cache)
        }
    }

    pub struct DeviceBufferList<'a> {
        buffers: Vec<Box<dyn DeviceDrop + 'a>>,
    }

    impl<'a> DeviceBufferList<'a> {
        pub fn new() -> Self {
            DeviceBufferList {
                buffers: Vec::new(),
            }
        }

        pub fn add<T: DeviceDrop + 'a>(&mut self, buffer: T) {
            self.buffers.push(Box::new(buffer));
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

    pub trait DeviceDrop {
        fn drop_async(self: Box<Self>, stream: &Stream) -> CudaResult<()>;
    }

    impl<T: DeviceCopy> DeviceDrop for DeviceBuffer<T> {
        fn drop_async(self: Box<Self>, stream: &Stream) -> CudaResult<()> {
            DeviceBuffer::drop_async(*self, stream)?;
            Ok(())
        }
    }

    impl<T: DeviceCopy> DeviceDrop for DeviceBox<T> {
        fn drop_async(self: Box<Self>, stream: &Stream) -> CudaResult<()> {
            DeviceBox::drop_async(*self, stream)?;
            Ok(())
        }
    }

    pub struct BuildResult<'a, T: Builder<'a>> {
        inner_device: T::Output,
        inner_host: T,
        stream: &'a Stream,
        buffers: DeviceBufferList<'a>,
        device_box: RefCell<Option<DeviceBox<T::Output>>>,
    }

    impl<'a, T: Builder<'a>> BuildResult<'a, T> {
        pub fn new(
            inner_device: T::Output,
            inner_host: T,
            stream: &'a Stream,
            buffers: DeviceBufferList<'a>,
        ) -> Self {
            BuildResult {
                inner_device,
                inner_host,
                stream,
                buffers,
                device_box: RefCell::new(None),
            }
        }

        pub fn split(self) -> (T::Output, T, DeviceBufferList<'a>) {
            (self.inner_device, self.inner_host, self.buffers)
        }

        pub unsafe fn as_device_ptr(&self) -> CudaResult<DevicePointer<T::Output>> {
            if self.device_box.borrow().is_none() {
                self.device_box
                    .replace(Some(DeviceBox::new_async(&self.inner_device, self.stream)?));
            }
            Ok(self
                .device_box
                .borrow()
                .as_ref()
                .expect("DeviceBox not initialized")
                .as_device_ptr())
        }

        pub fn copy_back(&mut self) -> CudaResult<&T> {
            if let Some(device_box) = self.device_box.borrow().as_ref() {
                self.inner_host.copy_back(device_box)?;
                Ok(&self.inner_host)
            } else {
                Err(CudaError::NotMappedAsPointer)
            }
        }
    }

    fn is_pointer<T: Any>() -> bool {
        let type_name = std::any::type_name::<T>();
        type_name.starts_with("*const ") || type_name.starts_with("*mut ")
    }

    impl<T: DeviceCopy + 'static> BuildResultType for T {}

    impl<'a, T: BuildResultType + Any> Builder<'a> for T {
        type Output = T;

        fn build_inner(self, _cache: &mut Cache<'_>) -> Self::Output {
            if is_pointer::<Self>() {
                panic!("Cannot build a pointer directly. Use a builder instead.");
            }
            self
        }

        unsafe fn build_device_inner(
            self,
            stream: &'a Stream,
            _cache: &mut Cache<'_>,
        ) -> CudaResult<BuildResult<'a, Self::Output>> {
            if is_pointer::<Self>() {
                panic!("Cannot build a pointer directly. Use a builder instead.");
            }
            Ok(BuildResult::new(
                self,
                self,
                stream,
                DeviceBufferList::new(),
            ))
        }

        fn copy_back(&mut self, device_box: &DeviceBox<Self::Output>) -> CudaResult<()> {
            if is_pointer::<Self>() {
                panic!("Cannot build a pointer directly. Use a builder instead.");
            }
            device_box.copy_to(self)?;
            Ok(())
        }
    }
}
