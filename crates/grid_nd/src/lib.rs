#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub use grid_nd_kernels::{GridND, GridViewND, GridViewNDMut};

#[cfg(test)]
mod tests {
    static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

    use std::error::Error;

    use super::*;
    use cust::prelude::*;

    #[test]
    fn cuda_grid_1d() -> Result<(), Box<dyn Error>> {
        let dims = [1024];

        // initialize CUDA, this will pick the first available device and will
        // make a CUDA context from it.
        // We don't need the context for anything but it must be kept alive.
        let _ctx = cust::quick_init()?;

        // Make the CUDA module, modules just house the GPU code for the kernels we created.
        // they can be made from PTX code, cubins, or fatbins.
        let module = Module::from_ptx(PTX, &[])?;

        // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
        // GPU calls.
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let lhs = GridND::<f32, 1>::new_random(dims);
        let lhs_device = lhs.to_device()?;
        let rhs = GridND::<f32, 1>::new_random(dims);
        let rhs_device = rhs.to_device()?;
        let mut result = GridND::<f32, 1>::new_zeroed(dims);
        let result_device = result.to_device()?;

        let vecadd = module.get_function("vecadd_1d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let grid_size = (dims[0] as u32).div_ceil(block_size);

        println!(
            "using {:?} blocks and {:?} threads per block",
            grid_size, block_size
        );

        // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
        // not block the thread until the kernel is finished.
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                vecadd<<<grid_size, block_size, 0, stream>>>(
                    lhs_device.as_device_ptr(),
                    rhs_device.as_device_ptr(),
                    result_device.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;

        result.copy_back(&result_device)?;

        // Verify the result
        for i in 0..dims[0] {
            assert_eq!(lhs.at(i) + rhs.at(i), *result.at(i));
        }

        Ok(())
    }

    #[test]
    fn cuda_grid_1d_async() -> Result<(), Box<dyn Error>> {
        let dims = [1024];

        // initialize CUDA, this will pick the first available device and will
        // make a CUDA context from it.
        // We don't need the context for anything but it must be kept alive.
        let _ctx = cust::quick_init()?;

        // Make the CUDA module, modules just house the GPU code for the kernels we created.
        // they can be made from PTX code, cubins, or fatbins.
        let module = Module::from_ptx(PTX, &[])?;

        // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
        // GPU calls.
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let lhs = GridND::<f32, 1>::new_random(dims);
        let lhs_device = unsafe { lhs.to_device_async(&stream)? };
        let rhs = GridND::<f32, 1>::new_random(dims);
        let rhs_device = unsafe { rhs.to_device_async(&stream)? };
        let mut result = GridND::<f32, 1>::new_zeroed(dims);
        let result_device = unsafe { result.to_device_async(&stream)? };

        let vecadd = module.get_function("vecadd_1d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let (grid_size, block_size) = lhs.grid_and_block_size(block_size);

        println!(
            "using {:?} blocks and {:?} threads per block",
            grid_size, block_size
        );

        // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
        // not block the thread until the kernel is finished.
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                vecadd<<<grid_size, block_size, 0, stream>>>(
                    lhs_device.as_device_ptr(),
                    rhs_device.as_device_ptr(),
                    result_device.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;

        result.copy_back(&result_device)?;

        // Verify the result
        for i in 0..dims[0] {
            assert_eq!(lhs.at(i) + rhs.at(i), *result.at(i));
        }

        Ok(())
    }

    #[test]
    fn cuda_grid_2d() -> Result<(), Box<dyn Error>> {
        let dims = [1024, 1024];

        // initialize CUDA, this will pick the first available device and will
        // make a CUDA context from it.
        // We don't need the context for anything but it must be kept alive.
        let _ctx = cust::quick_init()?;

        // Make the CUDA module, modules just house the GPU code for the kernels we created.
        // they can be made from PTX code, cubins, or fatbins.
        let module = Module::from_ptx(PTX, &[])?;

        // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
        // GPU calls.
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let lhs = GridND::<f32, 2>::new_random(dims);
        let lhs_device = lhs.to_device()?;
        let rhs = GridND::<f32, 2>::new_random(dims);
        let rhs_device = rhs.to_device()?;
        let mut result = GridND::<f32, 2>::new_zeroed(dims);
        let result_device = result.to_device()?;

        let vecadd = module.get_function("vecadd_2d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let (grid_size, block_size) = lhs.grid_and_block_size(block_size);

        println!(
            "using {:?} blocks and {:?} threads per block",
            grid_size, block_size
        );

        // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
        // not block the thread until the kernel is finished.
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                vecadd<<<grid_size, block_size, 0, stream>>>(
                    lhs_device.as_device_ptr(),
                    rhs_device.as_device_ptr(),
                    result_device.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;

        result.copy_back(&result_device)?;

        // Verify the result
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                assert_eq!(lhs.at(i).at(j) + rhs.at(i).at(j), *result.at(i).at(j));
            }
        }

        Ok(())
    }

    #[test]
    fn cuda_grid_2d_async() -> Result<(), Box<dyn Error>> {
        let dims = [1024, 1024];

        // initialize CUDA, this will pick the first available device and will
        // make a CUDA context from it.
        // We don't need the context for anything but it must be kept alive.
        let _ctx = cust::quick_init()?;

        // Make the CUDA module, modules just house the GPU code for the kernels we created.
        // they can be made from PTX code, cubins, or fatbins.
        let module = Module::from_ptx(PTX, &[])?;

        // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
        // GPU calls.
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let lhs = GridND::<f32, 2>::new_random(dims);
        let lhs_device = unsafe { lhs.to_device_async(&stream)? };
        let rhs = GridND::<f32, 2>::new_random(dims);
        let rhs_device = unsafe { rhs.to_device_async(&stream)? };
        let mut result = GridND::<f32, 2>::new_zeroed(dims);
        let result_device = unsafe { result.to_device_async(&stream)? };

        let vecadd = module.get_function("vecadd_2d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let (grid_size, block_size) = lhs.grid_and_block_size(block_size);

        println!(
            "using {:?} blocks and {:?} threads per block",
            grid_size, block_size
        );

        // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
        // not block the thread until the kernel is finished.
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                vecadd<<<grid_size, block_size, 0, stream>>>(
                    lhs_device.as_device_ptr(),
                    rhs_device.as_device_ptr(),
                    result_device.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;

        result.copy_back(&result_device)?;

        // Verify the result
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                assert_eq!(lhs.at(i).at(j) + rhs.at(i).at(j), *result.at(i).at(j));
            }
        }

        Ok(())
    }

    #[test]
    fn cuda_grid_3d() -> Result<(), Box<dyn Error>> {
        let dims = [128, 128, 128];

        // initialize CUDA, this will pick the first available device and will
        // make a CUDA context from it.
        // We don't need the context for anything but it must be kept alive.
        let _ctx = cust::quick_init()?;

        // Make the CUDA module, modules just house the GPU code for the kernels we created.
        // they can be made from PTX code, cubins, or fatbins.
        let module = Module::from_ptx(PTX, &[])?;

        // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
        // GPU calls.
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let lhs = GridND::<f32, 3>::new_random(dims);
        let lhs_device = lhs.to_device()?;
        let rhs = GridND::<f32, 3>::new_random(dims);
        let rhs_device = rhs.to_device()?;
        let mut result = GridND::<f32, 3>::new_zeroed(dims);
        let result_device = result.to_device()?;

        let vecadd = module.get_function("vecadd_3d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let (grid_size, block_size) = lhs.grid_and_block_size(block_size);

        println!(
            "using {:?} blocks and {:?} threads per block",
            grid_size, block_size
        );

        // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
        // not block the thread until the kernel is finished.
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                vecadd<<<grid_size, block_size, 0, stream>>>(
                    lhs_device.as_device_ptr(),
                    rhs_device.as_device_ptr(),
                    result_device.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;

        result.copy_back(&result_device)?;

        // Verify the result
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    assert_eq!(
                        lhs.at(i).at(j).at(k) + rhs.at(i).at(j).at(k),
                        *result.at(i).at(j).at(k)
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn cuda_grid_3d_async() -> Result<(), Box<dyn Error>> {
        let dims = [128, 128, 128];

        // initialize CUDA, this will pick the first available device and will
        // make a CUDA context from it.
        // We don't need the context for anything but it must be kept alive.
        let _ctx = cust::quick_init()?;

        // Make the CUDA module, modules just house the GPU code for the kernels we created.
        // they can be made from PTX code, cubins, or fatbins.
        let module = Module::from_ptx(PTX, &[])?;

        // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
        // GPU calls.
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let lhs = GridND::<f32, 3>::new_random(dims);
        let lhs_device = unsafe { lhs.to_device_async(&stream)? };
        let rhs = GridND::<f32, 3>::new_random(dims);
        let rhs_device = unsafe { rhs.to_device_async(&stream)? };
        let mut result = GridND::<f32, 3>::new_zeroed(dims);
        let result_device = unsafe { result.to_device_async(&stream)? };

        let vecadd = module.get_function("vecadd_3d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let (grid_size, block_size) = lhs.grid_and_block_size(block_size);

        println!(
            "using {:?} blocks and {:?} threads per block",
            grid_size, block_size
        );

        // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
        // not block the thread until the kernel is finished.
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                vecadd<<<grid_size, block_size, 0, stream>>>(
                    lhs_device.as_device_ptr(),
                    rhs_device.as_device_ptr(),
                    result_device.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;

        result.copy_back(&result_device)?;

        // Verify the result
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    assert_eq!(
                        lhs.at(i).at(j).at(k) + rhs.at(i).at(j).at(k),
                        *result.at(i).at(j).at(k)
                    );
                }
            }
        }

        Ok(())
    }
}
