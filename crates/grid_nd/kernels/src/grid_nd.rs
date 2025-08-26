use core::marker::PhantomData;

use crate::{Assert, IsTrue};

use gpu_builder::{BuildResultType, Builder, DeviceImpl};

#[repr(C)]
pub struct GridND<'a, T: Builder<'a>, const N: usize> {
    data: *mut T,
    dims: [usize; N],
    _marker: PhantomData<&'a T>,
}

#[repr(C)]
#[derive(DeviceImpl)]
#[builder(GridND)]
#[use_lifetime("'a")]
pub struct GridNDDevice<T: BuildResultType, const N: usize> {
    data: *mut T,
    dims: [usize; N],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GridViewND<'a, T, const N: usize> {
    data: *const T,
    dims: [usize; N],
    _marker: PhantomData<&'a T>,
}

#[repr(C)]
pub struct GridViewNDMut<'a, T, const N: usize> {
    data: *mut T,
    dims: [usize; N],
    _marker: PhantomData<&'a T>,
}

impl<'b, T: Builder<'b>> GridND<'b, T, 1> {
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

impl<'b, T: Builder<'b>, const N: usize> GridND<'b, T, N>
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
            _marker: PhantomData,
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
            _marker: PhantomData,
        }
    }
}

impl<'a, T> GridViewNDMut<'a, T, 1> {
    pub fn at(&self, index: usize) -> &'a T {
        assert!(index < self.dims[0], "Index out of bounds");
        // Safety: We assume index is within bounds and data is valid, we can dereference safely
        unsafe { &*self.data.add(index) }
    }

    pub fn at_mut(&'a mut self, index: usize) -> &'a mut T {
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
    pub fn at(&self, index: usize) -> GridViewND<'a, T, { N - 1 }> {
        assert!(index < self.dims[0], "Index out of bounds");
        let stride = self.dims[1..].iter().product::<usize>();
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(index * stride) };
        GridViewND {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _marker: PhantomData,
        }
    }

    pub fn at_mut(&'a mut self, index: usize) -> GridViewNDMut<'a, T, { N - 1 }> {
        assert!(index < self.dims[0], "Index out of bounds");
        let stride = self.dims[1..].iter().product::<usize>();
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(index * stride) };
        GridViewNDMut {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _marker: PhantomData,
        }
    }
}

impl<'a, T> GridViewND<'a, T, 1> {
    pub fn at(&self, index: usize) -> &'a T {
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
    pub fn at(&self, index: usize) -> GridViewND<'a, T, { N - 1 }> {
        assert!(index < self.dims[0], "Index out of bounds");
        let stride = self.dims[1..].iter().product::<usize>();
        // Safety: We assume index is within bounds and data is valid
        let data = unsafe { self.data.add(index * stride) };
        GridViewND {
            data: data,
            dims: self.dims[1..].try_into().unwrap(),
            _marker: PhantomData,
        }
    }
}

#[cfg(not(target_os = "cuda"))]
pub mod host_impls {
    use super::*;
    use core::cmp::min;
    use cust::{
        error::CudaResult,
        function::{BlockSize, GridSize},
        memory::{AsyncCopyDestination, DeviceBox},
        prelude::DeviceBuffer,
        stream::Stream,
    };
    use gpu_builder::{BuildResult, BuildResultType, Builder, Cache, DeviceBufferList};
    use rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    };

    impl<'a, T: Copy + Builder<'a>, const N: usize> GridND<'a, T, N> {
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
                _marker: PhantomData,
            }
        }
    }

    impl<'a, T: Copy + Default + Builder<'a>, const N: usize> GridND<'a, T, N> {
        /// Creates a GridND with heap-allocated zero-initialized buffer.
        pub fn new_zeroed(dims: [usize; N]) -> Self {
            Self::new(dims, T::default())
        }
    }

    impl<'a, T: Copy + Default + Builder<'a>, const N: usize> GridND<'a, T, N>
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

    impl<'a, T: Builder<'a>, const N: usize> Drop for GridND<'a, T, N> {
        fn drop(&mut self) {
            // Safety: We assume the data was allocated with Vec and we are responsible for freeing it.
            unsafe {
                let total_elems = self.dims.iter().product::<usize>();
                let _ = Vec::from_raw_parts(self.data, total_elems, total_elems);
            }
        }
    }

    impl<'a, T: BuildResultType, const N: usize> Builder<'a> for GridND<'a, T, N>
    where
        Self: 'a,
    {
        type Output = GridNDDevice<T, N>;
        fn build_inner(self, _cache: &mut Cache) -> GridNDDevice<T, N> {
            let result = GridNDDevice {
                data: self.data,
                dims: self.dims,
            };
            mem::forget(self); // Prevent double free
            result
        }

        unsafe fn build_device_inner(
            self,
            stream: &'a Stream,
            _cache: &mut Cache<'_>,
        ) -> CudaResult<BuildResult<'a, GridND<'a, T, N>>> {
            let total_elems = self.dims.iter().product::<usize>();
            let mut device_buffers = DeviceBufferList::new();
            // Safety: We assume the data is valid and we can copy it to the device
            // Safety: We make sure to initialize the DeviceBuffer correctly
            let device_buffer = unsafe {
                let mut device_buffer = DeviceBuffer::uninitialized_async(total_elems, stream)?;
                let data = std::slice::from_raw_parts(self.data, total_elems);
                device_buffer.async_copy_from(data, stream)?;
                device_buffer
            };
            let device_ptr = device_buffer.as_device_ptr().as_mut_ptr();
            device_buffers.add(device_buffer);
            let inner_host = GridND {
                data: self.data,
                dims: self.dims,
                _marker: PhantomData,
            };
            mem::forget(self); // Prevent double free
            Ok(BuildResult::new(
                GridNDDevice {
                    data: device_ptr,
                    dims: inner_host.dims,
                },
                inner_host,
                stream,
                device_buffers,
            ))
        }

        fn copy_back(
            &mut self,
            device_grid: &DeviceBox<GridNDDevice<<T as Builder<'a>>::Output, N>>,
        ) -> CudaResult<()> {
            let grid = device_grid.as_host_value()?;
            assert_eq!(grid.dims, self.dims, "Dimensions mismatch after copy back");
            let total_elems = self.dims.iter().product::<usize>();
            for i in 0..total_elems {
                let value: DeviceBox<<T as Builder<'a>>::Output> =
                    unsafe { DeviceBox::from_raw(grid.data.add(i) as u64) };
                // Safety: We assume the data is valid and we can write to it
                let data = unsafe { &mut *self.data.add(i) };
                data.copy_back(&value)?;
                mem::forget(value); // Prevent double free
            }
            Ok(())
        }
    }

    impl<'a, T: Builder<'a>> GridND<'a, T, 1> {
        pub fn grid_and_block_size(
            dims: [usize; 1],
            recommended_block_size: u32,
        ) -> (GridSize, BlockSize) {
            let mut block_size = recommended_block_size;
            if block_size > 32 {
                block_size = block_size.div_floor(32) * 32; // Ensure block size is a multiple of 32
            }

            let grid_size = (dims[0] as u32).div_ceil(block_size);

            (grid_size.into(), block_size.into())
        }
    }

    use std::{cmp::max, mem};
    impl<'a, T: Builder<'a>> GridND<'a, T, 2> {
        pub fn grid_and_block_size(
            dims: [usize; 2],
            recommended_block_size: u32,
        ) -> (GridSize, BlockSize) {
            let block_size_x = min(recommended_block_size, 32);
            let block_size_y = recommended_block_size.div_floor(block_size_x);
            let mut grid_size = [1u32; 2];
            let block_size = [block_size_x, block_size_y];

            for i in 0..2 {
                grid_size[i] = (dims[1 - i] as u32).div_ceil(block_size[i]);
            }
            let grid_size = (grid_size[0], grid_size[1]);
            let block_size = (block_size[0], block_size[1]);
            (grid_size.into(), block_size.into())
        }
    }

    impl<'a, T: Builder<'a>> GridND<'a, T, 3> {
        pub fn grid_and_block_size(
            dims: [usize; 3],
            recommended_block_size: u32,
        ) -> (GridSize, BlockSize) {
            let block_size_x = min(recommended_block_size, 32);
            let block_size_z = max(
                (recommended_block_size as f32 / block_size_x as f32)
                    .sqrt()
                    .floor() as u32,
                1,
            );
            let block_size_y = recommended_block_size.div_floor(block_size_x * block_size_z);
            let mut grid_size = [1u32; 3];
            let block_size = [block_size_x, block_size_y, block_size_z];

            for i in 0..3 {
                grid_size[i] = (dims[2 - i] as u32).div_ceil(block_size[i]);
            }
            let grid_size = (grid_size[0], grid_size[1], grid_size[2]);
            let block_size = (block_size[0], block_size[1], block_size[2]);
            (grid_size.into(), block_size.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

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

impl<'a, T: Builder<'a>, const N: usize> GridND<'a, T, N> {
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
    _marker: PhantomData<&'a T>,
}

impl<'a, 'b, T: Builder<'b>> IntoIterator for &'a GridND<'b, T, 1> {
    type Item = &'a T;
    type IntoIter = GridViewIter<'a, T, 1>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIter::<T, 1> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _marker: PhantomData,
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

impl<'a, 'b, T: Builder<'b>, const N: usize> IntoIterator for &'a GridND<'b, T, N>
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
            _marker: PhantomData,
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
            _marker: PhantomData,
        })
    }
}

pub struct GridViewIterMut<'a, T, const N: usize> {
    data: *mut T,
    dims: [usize; N],
    index: usize,
    stride: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, 'b, T: Builder<'b>> IntoIterator for &'a mut GridND<'b, T, 1> {
    type Item = &'a mut T;
    type IntoIter = GridViewIterMut<'a, T, 1>;

    fn into_iter(self) -> Self::IntoIter {
        let stride = self.dims[1..].iter().product::<usize>();

        GridViewIterMut::<T, 1> {
            data: self.data,
            dims: self.dims,
            index: 0,
            stride,
            _marker: PhantomData,
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

impl<'a, 'b, T: Builder<'b>, const N: usize> IntoIterator for &'a mut GridND<'b, T, N>
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
            _marker: PhantomData,
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
            _marker: PhantomData,
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
            _marker: PhantomData,
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
            _marker: PhantomData,
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
            _marker: PhantomData,
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
            _marker: PhantomData,
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
            _marker: PhantomData,
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
            _marker: PhantomData,
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

#[cfg(not(target_os = "cuda"))]
impl<'a, T: Builder<'a>, const N: usize> GridND<'a, T, N> {
    pub fn map<F, R: Builder<'a>>(self, mut f: F) -> GridND<'a, R, N>
    where
        F: FnMut(T) -> R,
    {
        let total_elems = self.dims.iter().product::<usize>();

        let data = unsafe { Vec::from_raw_parts(self.data, total_elems, total_elems) };
        let mut new_data: Vec<R> = Vec::with_capacity(total_elems);
        // Leak the Vec to keep memory stable and get a raw pointer
        let new_data_ptr = new_data.as_mut_ptr();

        // Don't drop the Vec — we're now managing memory manually
        core::mem::forget(new_data);

        // Safety: We assume the data is valid and we can modify it
        for (i, value) in data.into_iter().enumerate() {
            let new_data = unsafe { new_data_ptr.add(i) };
            // Apply the function to each element
            let new_value = f(value);
            // Safety: We assume data is valid and we can write to it
            unsafe { *new_data = new_value };
        }

        let dims = self.dims;
        core::mem::forget(self); // Prevent double free

        GridND {
            data: new_data_ptr,
            dims,
            _marker: PhantomData,
        }
    }
}

#[cfg(not(target_os = "cuda"))]
impl<'a, T: Clone + Builder<'a>, const N: usize> Clone for GridND<'a, T, N> {
    fn clone(&self) -> Self {
        let total_elems = self.dims.iter().product::<usize>();
        // Safety: We assume the data is valid and we can copy it
        let data = unsafe { Vec::from_raw_parts(self.data, total_elems, total_elems) };
        let mut new_data: Vec<T> = Vec::with_capacity(total_elems);
        for value in &data {
            new_data.push(value.clone());
        }
        // Leak the Vec to keep memory stable and get a raw pointer
        let new_data_ptr = new_data.as_mut_ptr();

        // Don't drop the Vec — we're now managing memory manually
        core::mem::forget(new_data);
        core::mem::forget(data); // Prevent double free

        GridND {
            data: new_data_ptr,
            dims: self.dims,
            _marker: PhantomData,
        }
    }
}

#[cfg(not(target_os = "cuda"))]
impl<'a, T: Builder<'a, Output = T> + BuildResultType, const N: usize> GridND<'a, Option<T>, N> {
    pub fn transpose(self) -> Option<GridND<'a, T, N>> {
        let total_elems = self.dims.iter().product::<usize>();
        let data = unsafe { Vec::from_raw_parts(self.data, total_elems, total_elems) };
        let mut new_data: Vec<T> = Vec::with_capacity(total_elems);
        // Leak the Vec to keep memory stable and get a raw pointer
        let new_data_ptr = new_data.as_mut_ptr();

        // Don't drop the Vec — we're now managing memory manually
        core::mem::forget(new_data);

        // Safety: We assume the data is valid and we can modify it
        for (i, value) in data.into_iter().enumerate() {
            if let Some(val) = value {
                let new_data = unsafe { new_data_ptr.add(i) };
                // Safety: We assume data is valid and we can write to it
                unsafe { *new_data = val };
            } else {
                return None; // If any value is None, we cannot transpose
            }
        }

        let dims = self.dims;
        core::mem::forget(self); // Prevent double free

        Some(GridND {
            data: new_data_ptr,
            dims,
            _marker: PhantomData,
        })
    }
}

#[cfg(not(target_os = "cuda"))]
impl<
    'a,
    T: Builder<'a, Output = T> + BuildResultType,
    const N: usize,
    E: Builder<'a, Output = E> + BuildResultType,
> GridND<'a, Result<T, E>, N>
{
    pub fn transpose(self) -> Result<GridND<'a, T, N>, E> {
        let total_elems = self.dims.iter().product::<usize>();
        let data = unsafe { Vec::from_raw_parts(self.data, total_elems, total_elems) };
        let mut new_data: Vec<T> = Vec::with_capacity(total_elems);
        // Leak the Vec to keep memory stable and get a raw pointer
        let new_data_ptr = new_data.as_mut_ptr();

        // Don't drop the Vec — we're now managing memory manually
        core::mem::forget(new_data);

        // Safety: We assume the data is valid and we can modify it
        for (i, value) in data.into_iter().enumerate() {
            match value {
                Ok(val) => {
                    let new_data = unsafe { new_data_ptr.add(i) };
                    // Safety: We assume data is valid and we can write to it
                    unsafe { *new_data = val };
                }
                Err(err) => return Err(err),
            }
        }

        let dims = self.dims;

        core::mem::forget(self);

        Ok(GridND {
            data: new_data_ptr,
            dims,
            _marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod map_tests {
    use super::*;

    #[test]
    #[allow(unused_mut)]
    fn test_gridnd_map() {
        let dims = [5, 5];
        let grid = GridND::<u32, 2>::new_zeroed(dims);

        // Map to double each value
        let mut grid = grid.map(|val| val * 2);

        for val in &grid {
            for inner_val in &val {
                assert_eq!(*inner_val, 0); // Since original was zeroed, doubled should also be zero
            }
        }

        // Modify original grid to have some values
        for mut val in &mut grid {
            for mut inner_val in &mut val {
                *inner_val = 1; // Set all values to 1
            }
        }

        let grid = grid.map(|val| val * 2);

        for val in &grid {
            for inner_val in &val {
                assert_eq!(*inner_val, 2); // Now doubled values should be 2
            }
        }
    }
}
