use core::marker::PhantomData;

use crate::{Assert, IsTrue};

#[repr(C)]
pub struct GridND<T, const N: usize> {
    data: *mut T,
    dims: [usize; N],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GridViewND<'a, T, const N: usize> {
    data: *const T,
    dims: [usize; N],
    _phantom: PhantomData<&'a T>,
}

#[repr(C)]
pub struct GridViewNDMut<'a, T, const N: usize> {
    data: *mut T,
    dims: [usize; N],
    _phantom: PhantomData<&'a T>,
}

impl<T> GridND<T, 1> {
    pub fn at<'a>(&'a self, index: usize) -> &'a T {
        assert!(index < self.dims[0], "Index out of bounds");
        // Safety: We assume index is within bounds and data is valid, we can dereference safely
        unsafe { &*self.data.add(index) }
    }
    pub fn at_mut<'a>(&'a mut self, index: usize) -> &'a mut T {
        assert!(index < self.dims[0], "Index out of bounds");
        // Safety: We assume index is within bounds and data is valid, we can dereference safely
        unsafe { &mut *self.data.add(index) }
    }
}

impl<T, const N: usize> GridND<T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    pub fn at<'a>(&'a self, index: usize) -> GridViewND<'a, T, { N - 1 }> {
        assert!(index < self.dims[0], "Index out of bounds");
        let stride = self.dims[1..].iter().product::<usize>();
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(index * stride) };
        GridViewND {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _phantom: PhantomData,
        }
    }
    pub fn at_mut<'a>(&'a mut self, index: usize) -> GridViewNDMut<'a, T, { N - 1 }> {
        assert!(index < self.dims[0], "Index out of bounds");
        let stride = self.dims[1..].iter().product::<usize>();
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(index * stride) };
        GridViewNDMut {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> GridViewNDMut<'a, T, 1> {
    pub fn at<'b>(&'b self, index: usize) -> &'b T {
        assert!(index < self.dims[0], "Index out of bounds");
        // Safety: We assume index is within bounds and data is valid, we can dereference safely
        unsafe { &*self.data.add(index) }
    }

    pub fn at_mut<'b>(&'b mut self, index: usize) -> &'b mut T {
        assert!(index < self.dims[0], "Index out of bounds");
        // Safety: We assume index is within bounds and data is valid, we can dereference safely
        unsafe { &mut *self.data.add(index) }
    }
}

impl<'a, T, const N: usize> GridViewNDMut<'a, T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    pub fn at<'b>(&'b self, index: usize) -> GridViewND<'b, T, { N - 1 }> {
        assert!(index < self.dims[0], "Index out of bounds");
        let stride = self.dims[1..].iter().product::<usize>();
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(index * stride) };
        GridViewND {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _phantom: PhantomData,
        }
    }

    pub fn at_mut<'b>(&'b mut self, index: usize) -> GridViewNDMut<'b, T, { N - 1 }> {
        assert!(index < self.dims[0], "Index out of bounds");
        let stride = self.dims[1..].iter().product::<usize>();
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(index * stride) };
        GridViewNDMut {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> GridViewND<'a, T, 1> {
    pub fn at<'b>(&'b self, index: usize) -> &'b T {
        assert!(index < self.dims[0], "Index out of bounds");
        // Safety: We assume index is within bounds and data is valid, we can dereference safely
        unsafe { &*self.data.add(index) }
    }
}

impl<'a, T, const N: usize> GridViewND<'a, T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    pub fn at<'b>(&'b self, index: usize) -> GridViewND<'b, T, { N - 1 }> {
        assert!(index < self.dims[0], "Index out of bounds");
        let stride = self.dims[1..].iter().product::<usize>();
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(index * stride) };
        GridViewND {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _phantom: PhantomData,
        }
    }
}

#[cfg(not(target_os = "cuda"))]
mod host_impls {
    use super::*;
    use cust::{
        function::{BlockSize, GridSize},
        memory::{AsyncCopyDestination, CopyDestination, DeviceBox, DeviceCopy},
        prelude::{DeviceBuffer, DevicePointer},
        stream::Stream,
    };
    use rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    };

    impl<T: Copy, const N: usize> GridND<T, N> {
        /// Creates a GridND with heap-allocated zero-initialized buffer.
        pub fn new(dims: [usize; N], value: T) -> Self {
            let total_elems = dims.iter().product::<usize>();

            // Create Vec<T> filled with the specified value
            let mut vec = vec![value; total_elems];

            // Leak the Vec to keep memory stable and get a raw pointer
            let data_ptr = vec.as_mut_ptr();

            // Don't drop the Vec — we're now managing memory manually
            std::mem::forget(vec);

            GridND {
                data: data_ptr,
                dims,
            }
        }
    }

    impl<T: Copy + Default, const N: usize> GridND<T, N> {
        /// Creates a GridND with heap-allocated zero-initialized buffer.
        pub fn new_zeroed(dims: [usize; N]) -> Self {
            Self::new(dims, T::default())
        }
    }

    impl<T: Copy + Default, const N: usize> GridND<T, N>
    where
        StandardUniform: Distribution<T>,
    {
        pub fn new_random(dims: [usize; N]) -> Self {
            let grid = Self::new_zeroed(dims);
            let total_elems = dims.iter().product::<usize>();
            // convert all the data into a slice so we can easily fill it with random values
            let data_slice = unsafe { std::slice::from_raw_parts_mut(grid.data, total_elems) };
            let mut rng = rand::rng();
            for val in data_slice {
                *val = rng.random();
            }
            grid
        }
    }

    impl<T, const N: usize> Drop for GridND<T, N> {
        fn drop(&mut self) {
            // Safety: We assume the data was allocated with Vec and we are responsible for freeing it.
            unsafe {
                let total_elems = self.dims.iter().product::<usize>();
                let _ = Vec::from_raw_parts(self.data, total_elems, total_elems);
            }
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct GridNDDeviceInner<T: DeviceCopy, const N: usize> {
        data: *mut T,
        dims: [usize; N],
    }

    pub struct GridNDDevice<T: DeviceCopy, const N: usize> {
        device_buffer: DeviceBuffer<T>,
        device_ptr: DeviceBox<GridNDDeviceInner<T, N>>,
    }

    // Safety: We implement DeviceCopy for GridNDDeviceInner to allow it to be used in CUDA kernels.
    // This is safe because GridNDDeviceInner is just a wrapper around GridND with the data moved to GPU memory.
    unsafe impl<T: DeviceCopy, const N: usize> DeviceCopy for GridNDDeviceInner<T, N> {}

    impl<T: DeviceCopy, const N: usize> GridND<T, N> {
        pub fn to_device(&self) -> Result<GridNDDevice<T, N>, Box<dyn std::error::Error>> {
            let total_elems = self.dims.iter().product::<usize>();
            // Safety: We assume the data is valid and we can copy it to the device
            // Safety: We make sure to initialize the DeviceBuffer correctly
            let device_buffer = unsafe {
                let mut device_buffer = DeviceBuffer::uninitialized(total_elems)?;
                let data = std::slice::from_raw_parts(self.data, total_elems);
                device_buffer.copy_from(data)?;
                device_buffer
            };
            let device_ptr = DeviceBox::new(&GridNDDeviceInner {
                data: device_buffer.as_device_ptr().as_mut_ptr(),
                dims: self.dims,
            })?;
            Ok(GridNDDevice {
                device_buffer,
                device_ptr,
            })
        }

        pub unsafe fn to_device_async(
            &self,
            stream: &Stream,
        ) -> Result<GridNDDevice<T, N>, Box<dyn std::error::Error>> {
            let total_elems = self.dims.iter().product::<usize>();
            // Safety: We assume the data is valid and we can copy it to the device
            // Safety: We make sure to initialize the DeviceBuffer correctly
            let device_buffer = unsafe {
                let mut device_buffer = DeviceBuffer::uninitialized_async(total_elems, stream)?;
                let data = std::slice::from_raw_parts(self.data, total_elems);
                device_buffer.async_copy_from(data, stream)?;
                device_buffer
            };
            let device_ptr = unsafe {
                DeviceBox::new_async(
                    &GridNDDeviceInner {
                        data: device_buffer.as_device_ptr().as_mut_ptr(),
                        dims: self.dims,
                    },
                    stream,
                )?
            };
            Ok(GridNDDevice {
                device_buffer,
                device_ptr,
            })
        }

        pub fn copy_back(
            &mut self,
            device_grid: &GridNDDevice<T, N>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let mut grid = GridNDDeviceInner {
                data: self.data,
                dims: self.dims,
            };
            device_grid.device_ptr.copy_to(&mut grid)?;
            assert_eq!(grid.dims, self.dims, "Dimensions mismatch after copy back");
            let total_elems = self.dims.iter().product::<usize>();
            let data = unsafe { std::slice::from_raw_parts_mut(self.data, total_elems) };
            device_grid.device_buffer.copy_to(data)?;
            Ok(())
        }
    }

    impl<T: DeviceCopy, const N: usize> GridNDDevice<T, N> {
        pub fn as_device_ptr(&self) -> DevicePointer<GridNDDeviceInner<T, N>> {
            self.device_ptr.as_device_ptr()
        }
    }

    impl<T> GridND<T, 1> {
        pub fn grid_and_block_size(&self, recommended_block_size: u32) -> (GridSize, BlockSize) {
            let block_size = recommended_block_size;

            let grid_size = (self.dims[0] as u32).div_ceil(block_size);

            (grid_size.into(), block_size.into())
        }
    }

    use std::cmp::max;
    impl<T> GridND<T, 2> {
        pub fn grid_and_block_size(&self, recommended_block_size: u32) -> (GridSize, BlockSize) {
            let recommended_block_size =
                max((recommended_block_size as f32).sqrt().floor() as u32, 1);
            let mut grid_size = [1u32; 2];
            let block_size = [recommended_block_size; 2];

            for i in 0..2 {
                grid_size[i] = (self.dims[i] as u32).div_ceil(block_size[i]);
            }
            let grid_size = (grid_size[0], grid_size[1]);
            let block_size = (block_size[0], block_size[1]);
            (grid_size.into(), block_size.into())
        }
    }

    impl<T> GridND<T, 3> {
        pub fn grid_and_block_size(&self, recommended_block_size: u32) -> (GridSize, BlockSize) {
            let recommended_block_size =
                max((recommended_block_size as f32).cbrt().floor() as u32, 1);
            let mut grid_size = [1u32; 3];
            let block_size = [recommended_block_size; 3];

            for i in 0..3 {
                grid_size[i] = (self.dims[i] as u32).div_ceil(block_size[i]);
            }
            let grid_size = (grid_size[0], grid_size[1], grid_size[2]);
            let block_size = (block_size[0], block_size[1], block_size[2]);
            (grid_size.into(), block_size.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::mem;

    use super::*;

    #[test]
    fn gridnd_device_is_mirror() {
        assert_eq!(
            mem::size_of::<GridND<f32, 3>>(),
            mem::size_of::<host_impls::GridNDDeviceInner<f32, 3>>(),
            "Size mismatch between host and device types"
        );

        assert_eq!(
            mem::align_of::<GridND<f32, 3>>(),
            mem::align_of::<host_impls::GridNDDeviceInner<f32, 3>>(),
            "Alignment mismatch"
        );
    }

    #[test]
    fn gridnd_zeroed_fills_with_zeros() {
        let dims = [4, 4];
        let grid = GridND::<u32, 2>::new_zeroed(dims);
        let total = dims.iter().product::<usize>();

        // Safety: we're reading back data we just initialized
        for i in 0..total {
            let val = unsafe { *grid.data.add(i) };
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn gridnd_indexing_1d_works() {
        let dims = [10];
        let mut grid = GridND::<u32, 1>::new_zeroed(dims);

        assert_eq!(*grid.at(3), 0);
        assert_eq!(*grid.at_mut(3), 0);

        *grid.at_mut(3) = 42;

        assert_eq!(*grid.at(3), 42);
        assert_eq!(*grid.at_mut(3), 42);
    }

    #[test]
    fn gridnd_indexing_2d_works() {
        let dims = [10, 10];
        let mut grid = GridND::<u32, 2>::new_zeroed(dims);

        assert_eq!(*grid.at(3).at(3), 0);
        assert_eq!(*grid.at_mut(3).at(3), 0);
        assert_eq!(*grid.at_mut(3).at_mut(3), 0);

        *grid.at_mut(3).at_mut(3) = 42;

        assert_eq!(*grid.at(3).at(3), 42);
        assert_eq!(*grid.at_mut(3).at(3), 42);
        assert_eq!(*grid.at_mut(3).at_mut(3), 42);
    }

    #[test]
    fn gridnd_indexing_nd_works() {
        let dims = [10, 10, 10];
        let mut grid = GridND::<u32, 3>::new_zeroed(dims);

        assert_eq!(*grid.at(3).at(3).at(3), 0);
        assert_eq!(*grid.at_mut(3).at(3).at(3), 0);
        assert_eq!(*grid.at_mut(3).at_mut(3).at(3), 0);
        assert_eq!(*grid.at_mut(3).at_mut(3).at_mut(3), 0);

        *grid.at_mut(3).at_mut(3).at_mut(3) = 42;

        assert_eq!(*grid.at(3).at(3).at(3), 42);
        assert_eq!(*grid.at_mut(3).at(3).at(3), 42);
        assert_eq!(*grid.at_mut(3).at_mut(3).at(3), 42);
        assert_eq!(*grid.at_mut(3).at_mut(3).at_mut(3), 42);
    }

    #[test]
    fn gridnd_indexing_out_of_bounds_1d_panics() {
        let dims = [10];
        let mut grid = GridND::<u32, 1>::new_zeroed(dims);

        grid.at(9);
        grid.at_mut(9);
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            grid.at(10);
        });
        assert!(result.is_err());
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            let mut grid = GridND::<u32, 1>::new_zeroed(dims);
            grid.at_mut(10);
        });
        assert!(result.is_err());
    }

    #[test]
    fn gridnd_indexing_out_of_bounds_2d_panics() {
        let dims = [10, 10];
        let mut grid = GridND::<u32, 2>::new_zeroed(dims);

        grid.at(9).at(9);
        grid.at_mut(9).at(9);
        grid.at_mut(9).at_mut(9);
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            grid.at(9).at(10);
        });
        assert!(result.is_err());
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            let mut grid = GridND::<u32, 2>::new_zeroed(dims);
            grid.at_mut(9).at(10);
        });
        assert!(result.is_err());
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            let mut grid = GridND::<u32, 2>::new_zeroed(dims);
            grid.at_mut(9).at_mut(10);
        });
        assert!(result.is_err());
    }

    #[test]
    fn gridnd_indexing_out_of_bounds_nd_panics() {
        let dims = [10, 10, 10];
        let mut grid = GridND::<u32, 3>::new_zeroed(dims);

        grid.at(9).at(9).at(9);
        grid.at_mut(9).at(9).at(9);
        grid.at_mut(9).at_mut(9).at(9);
        grid.at_mut(9).at_mut(9).at_mut(9);
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            grid.at(9).at(9).at(10);
        });
        assert!(result.is_err());
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            let mut grid = GridND::<u32, 3>::new_zeroed(dims);
            grid.at_mut(9).at(9).at(10);
        });
        assert!(result.is_err());
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            let mut grid = GridND::<u32, 3>::new_zeroed(dims);
            grid.at_mut(9).at_mut(9).at(10);
        });
        assert!(result.is_err());
        // This should panic because index is out of bounds
        let result: Result<(), Box<dyn Any + Send + 'static>> = std::panic::catch_unwind(|| {
            let mut grid = GridND::<u32, 3>::new_zeroed(dims);
            grid.at_mut(9).at_mut(9).at_mut(10);
        });
        assert!(result.is_err());
    }
}

impl<T, const N: usize> GridND<T, N> {
    pub fn shape(&self) -> [usize; N] {
        self.dims
    }
}

impl<T, const N: usize> GridViewND<'_, T, N> {
    pub fn shape(&self) -> [usize; N] {
        self.dims
    }
}

impl<T, const N: usize> GridViewNDMut<'_, T, N> {
    pub fn shape(&self) -> [usize; N] {
        self.dims
    }
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    #[test]
    fn test_gridnd_shape() {
        let grid = GridND::<u32, 2>::new_zeroed([10, 10]);
        assert_eq!(grid.shape(), [10, 10]);
    }

    #[test]
    fn test_gridviewnd_shape() {
        let grid = GridND::<u32, 3>::new_zeroed([10, 10, 10]);
        let view = grid.at(0);
        assert_eq!(view.shape(), [10, 10]);
    }

    #[test]
    fn test_gridviewndmut_shape() {
        let mut grid = GridND::<u32, 3>::new_zeroed([10, 10, 10]);
        let view = grid.at_mut(0);
        assert_eq!(view.shape(), [10, 10]);
    }
}

// iterators
pub struct GridViewIter<'a, T, const N: usize> {
    data: *const T,
    dims: [usize; N],
    index: usize,
    stride: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> IntoIterator for &'a GridND<T, 1> {
    type Item = &'a T;
    type IntoIter = GridViewIter<'a, T, 1>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIter::<T, 1> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: 'a> Iterator for GridViewIter<'a, T, 1> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dims[0] {
            return None;
        }
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(self.index * self.stride) };
        self.index += 1;
        // Safety: We assume data is valid and we can dereference safely
        Some(unsafe { &*data })
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a GridND<T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    type Item = GridViewND<'a, T, { N - 1 }>;
    type IntoIter = GridViewIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIter::<T, N> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, const N: usize> Iterator for GridViewIter<'a, T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    type Item = GridViewND<'a, T, { N - 1 }>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dims[0] {
            return None;
        }
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(self.index * self.stride) };
        self.index += 1;
        // Safety: We assume data is valid and we can dereference safely
        Some(GridViewND {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _phantom: PhantomData,
        })
    }
}

pub struct GridViewIterMut<'a, T, const N: usize> {
    data: *mut T,
    dims: [usize; N],
    index: usize,
    stride: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> IntoIterator for &'a mut GridND<T, 1> {
    type Item = &'a mut T;
    type IntoIter = GridViewIterMut<'a, T, 1>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIterMut::<T, 1> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: 'a> Iterator for GridViewIterMut<'a, T, 1> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dims[0] {
            return None;
        }
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(self.index * self.stride) };
        self.index += 1;
        // Safety: We assume data is valid and we can dereference safely
        Some(unsafe { &mut *data })
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut GridND<T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    type Item = GridViewNDMut<'a, T, { N - 1 }>;
    type IntoIter = GridViewIterMut<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIterMut::<T, N> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, const N: usize> Iterator for GridViewIterMut<'a, T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    type Item = GridViewNDMut<'a, T, { N - 1 }>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dims[0] {
            return None;
        }
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(self.index * self.stride) };
        self.index += 1;
        // Safety: We assume data is valid and we can dereference safely
        Some(GridViewNDMut {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _phantom: PhantomData,
        })
    }
}

impl<'a, T> IntoIterator for &'a GridViewND<'a, T, 1> {
    type Item = &'a T;
    type IntoIter = GridViewIter<'a, T, 1>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIter::<T, 1> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a GridViewND<'a, T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    type Item = GridViewND<'a, T, { N - 1 }>;
    type IntoIter = GridViewIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIter::<T, N> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> IntoIterator for &'a GridViewNDMut<'a, T, 1> {
    type Item = &'a T;
    type IntoIter = GridViewIter<'a, T, 1>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIter::<T, 1> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a GridViewNDMut<'a, T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    type Item = GridViewND<'a, T, { N - 1 }>;
    type IntoIter = GridViewIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIter::<T, N> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> IntoIterator for &'a mut GridViewNDMut<'a, T, 1> {
    type Item = &'a mut T;
    type IntoIter = GridViewIterMut<'a, T, 1>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIterMut::<T, 1> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut GridViewNDMut<'a, T, N>
where
    [(); N - 1]:,
    Assert<{ N > 1 }>: IsTrue,
{
    type Item = GridViewNDMut<'a, T, { N - 1 }>;
    type IntoIter = GridViewIterMut<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIterMut::<T, N> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod iterator_tests {
    use super::*;

    #[test]
    fn test_1d() {
        let dims = [5];
        let mut grid = GridND::<u32, 1>::new_zeroed(dims);
        for val in &grid {
            assert_eq!(*val, 0); // Assuming data is initialized to 0
        }

        #[allow(unused_mut)]
        for mut val in &mut grid {
            *val = 42; // Modify the value
        }

        for val in &grid {
            assert_eq!(*val, 42); // Check if the value was modified
        }

        let sum = grid.into_iter().sum::<u32>();
        assert_eq!(sum, 42 * 5); // Check if the sum is correct
    }

    #[test]
    fn test_2d() {
        let dims = [5, 5];
        let mut grid = GridND::<u32, 2>::new_zeroed(dims);
        for val in &grid.at(2) {
            assert_eq!(*val, 0); // Assuming data is initialized to 0
        }
        for val in &grid.at_mut(2) {
            assert_eq!(*val, 0); // Assuming data is initialized to 0
        }

        for mut val in &mut grid {
            for inner_val in &mut val {
                *inner_val = 1; // Modify the value
            }
        }

        for val in &mut grid {
            for inner_val in &val {
                assert_eq!(*inner_val, 1); // Check if the value was modified
            }
        }

        for val in &mut grid.at_mut(2) {
            *val = 42; // Modify the value
        }

        let sum = grid
            .into_iter()
            .fold(0, |acc, val| acc + val.into_iter().sum::<u32>());
        assert_eq!(sum, 42 * 5 + 1 * 5 * 4); // Check if the sum is correct
    }

    #[test]
    fn test_3d() {
        let dims = [5, 5, 5];
        let mut grid = GridND::<u32, 3>::new_zeroed(dims);
        for val in &grid.at(2).at(2) {
            assert_eq!(*val, 0); // Assuming data is initialized to 0
        }
        for val in &grid.at_mut(2).at_mut(2) {
            assert_eq!(*val, 0); // Assuming data is initialized to 0
        }

        for mut val in &mut grid {
            for mut inner_val in &mut val {
                for inner_inner_val in &mut inner_val {
                    *inner_inner_val = 1; // Modify the value
                }
            }
        }

        for val in &mut grid {
            for inner_val in &val {
                for inner_inner_val in &inner_val {
                    assert_eq!(*inner_inner_val, 1); // Check if the value was modified
                }
            }
        }

        for val in &mut grid.at_mut(2).at_mut(2) {
            *val = 42; // Modify the value
        }

        let sum = grid.into_iter().fold(0, |acc, val| {
            acc + val
                .into_iter()
                .fold(0, |acc, inner_val| acc + inner_val.into_iter().sum::<u32>())
        });
        assert_eq!(sum, 42 * 5 + 1 * 5 * 4 + 1 * 5 * 5 * 4); // Check if the sum is correct
    }
}
