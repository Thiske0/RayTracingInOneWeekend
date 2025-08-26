#![feature(offset_of_enum)]

mod device_struct_struct {
    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::DeviceStruct)]
    struct A<'a, T: gpu_builder::Builder<'a>>
    where
        T: cust::memory::DeviceCopy,
    {
        x: B,
        y: &'a B,
        z: C<'a, T>,
        w: T,
    }

    #[allow(unused)]
    #[derive(Clone, Copy, cust::memory::DeviceCopy)]
    struct B {
        z: i32,
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::DeviceStruct)]
    struct C<'a, T: gpu_builder::Builder<'a>> {
        z: T,
        _marker: core::marker::PhantomData<&'a ()>,
    }

    impl<'a, T: gpu_builder::Builder<'a>> gpu_builder::Builder<'a> for C<'a, T> {
        type Output = CDevice<T::Output>;

        fn build_inner(self, cache: &mut gpu_builder::Cache<'a>) -> Self::Output {
            CDevice {
                z: self.z.build_inner(cache),
            }
        }

        unsafe fn build_device_inner(
            self,
            stream: &'a cust::stream::Stream,
            cache: &mut gpu_builder::Cache<'a>,
        ) -> cust::error::CudaResult<gpu_builder::BuildResult<'a, Self>> {
            let (z_device, z_host, z_buffers) = self.z.build_device_inner(stream, cache)?.split();
            Ok(gpu_builder::BuildResult::new(
                CDevice { z: z_device },
                C {
                    z: z_host,
                    _marker: core::marker::PhantomData,
                },
                stream,
                z_buffers,
            ))
        }

        fn copy_back(
            &mut self,
            c_device: &cust::memory::DeviceBox<<Self as gpu_builder::Builder<'a>>::Output>,
        ) -> Result<(), cust::error::CudaError> {
            let base_ptr = c_device.as_device_ptr().as_raw();

            let field_ptr = base_ptr + std::mem::offset_of!(Self, z) as u64;
            let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
            self.z.copy_back(&field_device)?;
            Ok(())
        }
    }

    impl<'a> From<&'a *const B> for &'a &B {
        fn from(value: &'a *const B) -> Self {
            unsafe { &*(value as *const *const B as *const Self) }
        }
    }

    impl<'a> From<&'a mut *const B> for &'a mut &B {
        fn from(value: &'a mut *const B) -> Self {
            unsafe { &mut *(value as *mut *const B as *mut Self) }
        }
    }

    impl<'a> gpu_builder::Builder<'a> for &'a B {
        type Output = *const B;

        fn build_inner(self, _cache: &mut gpu_builder::Cache<'a>) -> Self::Output {
            self
        }

        unsafe fn build_device_inner(
            self,
            stream: &'a cust::stream::Stream,
            cache: &mut gpu_builder::Cache<'a>,
        ) -> cust::error::CudaResult<gpu_builder::BuildResult<'a, Self>> {
            let (device, _host, mut buffers) = (*self).build_device_inner(stream, cache)?.split();
            let device_slice = std::slice::from_ref(&device);
            let device_buffer = cust::memory::DeviceBuffer::from_slice_async(device_slice, stream)?;
            let device_ptr = device_buffer.as_device_ptr().as_ptr();
            buffers.add(device_buffer);
            Ok(gpu_builder::BuildResult::new(
                device_ptr, self, stream, buffers,
            ))
        }

        fn copy_back(
            &mut self,
            b_device: &cust::memory::DeviceBox<Self::Output>,
        ) -> Result<(), cust::error::CudaError> {
            let field_device =
                unsafe { cust::memory::DeviceBox::from_raw(b_device.as_host_value()? as u64) };
            (*self).copy_back(&field_device)
        }
    }
}

mod builder_struct {
    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::Builder)]
    #[use_lifetime("'a")]
    struct D<'a, T: gpu_builder::Builder<'a>>
    where
        T: cust::memory::DeviceCopy,
    {
        x: E,
        y: &'a E,
        z: F<'a, T>,
        w: T,
    }

    #[allow(unused)]
    #[derive(Clone, Copy, cust::memory::DeviceCopy)]
    struct E {
        z: i32,
    }

    #[allow(unused)]
    struct F<'a, T: gpu_builder::Builder<'a>> {
        z: T,
        _marker: core::marker::PhantomData<&'a ()>,
    }

    struct FDevice<T: gpu_builder::BuildResultType> {
        y: T,
    }

    unsafe impl<'a, T: gpu_builder::BuildResultType> cust::memory::DeviceCopy for FDevice<T> {}
    impl<'a, T: gpu_builder::BuildResultType> Copy for FDevice<T> {}
    impl<'a, T: gpu_builder::BuildResultType> Clone for FDevice<T> {
        fn clone(&self) -> Self {
            FDevice { y: self.y.clone() }
        }
    }

    impl<'a, T: gpu_builder::Builder<'a>> gpu_builder::Builder<'a> for F<'a, T> {
        type Output = FDevice<T::Output>;

        fn build_inner(self, cache: &mut gpu_builder::Cache<'a>) -> Self::Output {
            FDevice {
                y: self.z.build_inner(cache),
            }
        }

        unsafe fn build_device_inner(
            self,
            stream: &'a cust::stream::Stream,
            cache: &mut gpu_builder::Cache<'a>,
        ) -> cust::error::CudaResult<gpu_builder::BuildResult<'a, Self>> {
            let (z_device, z_host, z_buffers) = self.z.build_device_inner(stream, cache)?.split();
            Ok(gpu_builder::BuildResult::new(
                FDevice { y: z_device },
                F {
                    z: z_host,
                    _marker: core::marker::PhantomData,
                },
                stream,
                z_buffers,
            ))
        }

        fn copy_back(
            &mut self,
            c_device: &cust::memory::DeviceBox<<Self as gpu_builder::Builder<'a>>::Output>,
        ) -> Result<(), cust::error::CudaError> {
            let base_ptr = c_device.as_device_ptr().as_raw();

            let field_ptr = base_ptr + std::mem::offset_of!(Self, z) as u64;
            let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
            self.z.copy_back(&field_device)?;
            Ok(())
        }
    }

    impl<'a> gpu_builder::Builder<'a> for &'a E {
        type Output = *const E;

        fn build_inner(self, _cache: &mut gpu_builder::Cache<'a>) -> Self::Output {
            self
        }

        unsafe fn build_device_inner(
            self,
            stream: &'a cust::stream::Stream,
            cache: &mut gpu_builder::Cache<'a>,
        ) -> cust::error::CudaResult<gpu_builder::BuildResult<'a, Self>> {
            let (device, _host, mut buffers) = (*self).build_device_inner(stream, cache)?.split();
            let device_slice = std::slice::from_ref(&device);
            let device_buffer = cust::memory::DeviceBuffer::from_slice_async(device_slice, stream)?;
            let device_ptr = device_buffer.as_device_ptr().as_ptr();
            buffers.add(device_buffer);
            Ok(gpu_builder::BuildResult::new(
                device_ptr, self, stream, buffers,
            ))
        }

        fn copy_back(
            &mut self,
            b_device: &cust::memory::DeviceBox<<Self as gpu_builder::Builder<'a>>::Output>,
        ) -> Result<(), cust::error::CudaError> {
            let field_device =
                unsafe { cust::memory::DeviceBox::from_raw(b_device.as_host_value()? as u64) };
            (*self).copy_back(&field_device)
        }
    }
}
mod device_struct_enum {
    #[repr(C)]
    #[derive(gpu_builder::Builder)]
    #[use_lifetime("'a")]
    struct Wrapper<'a, T: gpu_builder::Builder<'a>> {
        x: T,
        _marker: core::marker::PhantomData<&'a ()>,
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::DeviceStruct)]
    enum Enum1<'a, T: gpu_builder::Builder<'a>> {
        T(Wrapper<'a, T>),
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::DeviceStruct)]
    enum Enum2 {
        T = 3,
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::DeviceStruct)]
    enum Enum3<'a, T: gpu_builder::Builder<'a>> {
        Quit,
        Move { x: Wrapper<'a, T>, y: i32 }, // Struct variant
        Write(bool),
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::DeviceStruct)]
    enum Enum4 {
        A(i32, i32),
        B(i32),
        C(),
    }

    impl<'a> gpu_builder::Builder<'a> for Enum4 {
        type Output = Enum4Device;

        fn build_inner(self, cache: &mut gpu_builder::Cache<'a>) -> Self::Output {
            match self {
                Enum4::A(x, y) => Enum4Device::A(x.build_inner(cache), y.build_inner(cache)),
                Enum4::B(x) => Enum4Device::B(x.build_inner(cache)),
                Enum4::C() => Enum4Device::C(),
            }
        }

        unsafe fn build_device_inner(
            self,
            stream: &'a cust::stream::Stream,
            cache: &mut gpu_builder::Cache<'a>,
        ) -> Result<gpu_builder::BuildResult<'a, Self>, cust::error::CudaError> {
            match self {
                Enum4::A(x, y) => {
                    let (x_device, x_host, mut x_buffers) =
                        x.build_device_inner(stream, cache)?.split();
                    let (y_device, y_host, y_buffers) =
                        y.build_device_inner(stream, cache)?.split();
                    x_buffers.combine(y_buffers);
                    Ok(gpu_builder::BuildResult::new(
                        Enum4Device::A(x_device, y_device),
                        Enum4::A(x_host, y_host),
                        stream,
                        x_buffers,
                    ))
                }
                Enum4::B(x) => {
                    let (x_device, x_host, x_buffers) =
                        x.build_device_inner(stream, cache)?.split();
                    Ok(gpu_builder::BuildResult::new(
                        Enum4Device::B(x_device),
                        Enum4::B(x_host),
                        stream,
                        x_buffers,
                    ))
                }
                Enum4::C() => Ok(gpu_builder::BuildResult::new(
                    Enum4Device::C(),
                    Enum4::C(),
                    stream,
                    gpu_builder::DeviceBufferList::new(),
                )),
            }
        }

        fn copy_back(
            &mut self,
            device_box: &cust::memory::DeviceBox<Self::Output>,
        ) -> cust::error::CudaResult<()> {
            let host_value = device_box.as_host_value()?;
            let base_ptr = device_box.as_device_ptr().as_raw();
            match host_value {
                Enum4Device::A(mut x, mut y) => {
                    let field_ptr = base_ptr + std::mem::offset_of!(Enum4Device, A.0) as u64;
                    let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                    x.copy_back(&field_device)?;

                    let field_ptr = base_ptr + std::mem::offset_of!(Enum4Device, A.1) as u64;
                    let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                    y.copy_back(&field_device)?;

                    *self = Enum4::A(x, y);
                }
                Enum4Device::B(mut x) => {
                    let field_ptr = base_ptr + std::mem::offset_of!(Enum4Device, A.0) as u64;
                    let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                    x.copy_back(&field_device)?;

                    *self = Enum4::B(x);
                }
                Enum4Device::C() => {
                    *self = Enum4::C();
                }
            }
            Ok(())
        }
    }
}

mod builder_enum {
    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::Builder)]
    #[use_lifetime("'a")]
    struct Wrapper<'a, T: gpu_builder::Builder<'a>> {
        x: T,
        _marker: core::marker::PhantomData<&'a ()>,
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::Builder)]
    #[use_lifetime("'a")]
    enum Enum1<'a, T: gpu_builder::Builder<'a>> {
        T(Wrapper<'a, T>),
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::Builder)]
    enum Enum2 {
        T = 3,
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::Builder)]
    #[use_lifetime("'a")]
    enum Enum3<'a, T: gpu_builder::Builder<'a>> {
        Quit,
        Move { x: Wrapper<'a, T>, y: i32 }, // Struct variant
        Write(bool),
    }

    #[allow(unused)]
    #[repr(C)]
    #[derive(gpu_builder::Builder)]
    enum Enum4 {
        A(i32, i32),
        B(i32),
        C(),
    }
}

#[cfg(test)]
mod test {
    use gpu_builder::{Builder, NiceBuilder};

    #[repr(C)]
    #[derive(Builder, PartialEq, Debug, Clone)]
    #[use_lifetime("'a")]
    struct TestStruct<'a, T: Builder<'a>> {
        x: T,
        _marker: core::marker::PhantomData<&'a ()>,
    }

    #[test]
    fn test_device_struct() {
        let test_struct = TestStruct {
            x: 42,
            _marker: core::marker::PhantomData,
        };
        let _result = test_struct.build();

        match std::panic::catch_unwind(|| {
            let test_struct = TestStruct {
                x: &42 as *const _,
                _marker: core::marker::PhantomData,
            };
            let _result = test_struct.clone().build();
        }) {
            Ok(_) => panic!("Expected panic, but expression executed successfully"),
            Err(_) => (), // Expected behavior - panic occurred
        }

        match std::panic::catch_unwind(|| {
            let test_struct = TestStruct {
                x: &mut 42 as *mut _,
                _marker: core::marker::PhantomData,
            };
            let _result = test_struct.clone().build();
        }) {
            Ok(_) => panic!("Expected panic, but expression executed successfully"),
            Err(_) => (), // Expected behavior - panic occurred
        }
    }
}
