#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub use grid_nd_kernels::{GridND, GridNDDevice, GridViewND, GridViewNDMut};

#[cfg(test)]
mod tests {
    static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

    use std::error::Error;

    use super::*;
    use cust::prelude::*;
    use gpu_builder::NiceBuilder;

    #[test]
    fn verify_allignment() {
        assert!(size_of::<GridND<f32, 1>>() == size_of::<GridNDDevice<f32, 1>>());
        assert!(align_of::<GridND<f32, 1>>() == align_of::<GridNDDevice<f32, 1>>());
    }

    #[test]
    fn cuda_grid_1d() -> Result<(), Box<dyn Error>> {
        let dims = [1023];

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
        let lhs_device = unsafe { lhs.clone().build_device(&stream)? };
        let rhs = GridND::<f32, 1>::new_random(dims);
        let rhs_device = unsafe { rhs.clone().build_device(&stream)? };
        let result = GridND::<f32, 1>::new_zeroed(dims);
        let mut result_device = unsafe { result.clone().build_device(&stream)? };

        let vecadd = module.get_function("vecadd_1d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let (grid_size, block_size) = GridND::<f32, 1>::grid_and_block_size(dims, block_size);

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
                    lhs_device.as_device_ptr()?,
                    rhs_device.as_device_ptr()?,
                    result_device.as_device_ptr()?,
                )
            )?;
        }

        stream.synchronize()?;

        let result = result_device.copy_back()?;

        // Verify the result
        for i in 0..dims[0] {
            assert_eq!(lhs.at(i) + rhs.at(i), *result.at(i));
        }

        Ok(())
    }

    #[test]
    fn cuda_grid_2d() -> Result<(), Box<dyn Error>> {
        let dims = [127, 63];

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
        let lhs_device = unsafe { lhs.clone().build_device(&stream)? };
        let rhs = GridND::<f32, 2>::new_random(dims);
        let rhs_device = unsafe { rhs.clone().build_device(&stream)? };
        let result = GridND::<f32, 2>::new_zeroed(dims);
        let mut result_device = unsafe { result.clone().build_device(&stream)? };

        let vecadd = module.get_function("vecadd_2d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let (grid_size, block_size) = GridND::<f32, 2>::grid_and_block_size(dims, block_size);

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
                    lhs_device.as_device_ptr()?,
                    rhs_device.as_device_ptr()?,
                    result_device.as_device_ptr()?,
                )
            )?;
        }

        stream.synchronize()?;

        let result = result_device.copy_back()?;

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
        let dims = [127, 31, 63];

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
        let lhs_device = unsafe { lhs.clone().build_device(&stream)? };
        let rhs = GridND::<f32, 3>::new_random(dims);
        let rhs_device = unsafe { rhs.clone().build_device(&stream)? };
        let result = GridND::<f32, 3>::new_zeroed(dims);
        let mut result_device = unsafe { result.clone().build_device(&stream)? };

        let vecadd = module.get_function("vecadd_3d_f32")?;

        // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
        // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
        // current CUDA device/architecture.
        let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

        let (grid_size, block_size) = GridND::<f32, 3>::grid_and_block_size(dims, block_size);

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
                    lhs_device.as_device_ptr()?,
                    rhs_device.as_device_ptr()?,
                    result_device.as_device_ptr()?,
                )
            )?;
        }

        stream.synchronize()?;

        let result = result_device.copy_back()?;

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
