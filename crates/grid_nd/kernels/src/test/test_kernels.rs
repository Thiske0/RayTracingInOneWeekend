use crate::{GridND, GridNDDevice, TupleConversion2D, TupleConversion3D};
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd_1d_f32(
    a: &GridNDDevice<f32, 1>,
    b: &GridNDDevice<f32, 1>,
    c: *mut GridNDDevice<f32, 1>,
) {
    let a: &GridND<'static, f32, 1> = a.into();
    let b: &GridND<'static, f32, 1> = b.into();
    // Safety: 'c' must point to a valid GridNDDevice<f32, 1> that is mutable.
    let c: &mut GridND<_, _> = unsafe { &mut *c }.into();
    let idx = thread::index_1d() as usize;
    let dims = a.shape();
    assert!(dims == b.shape() && dims == c.shape());
    if idx < dims[0] {
        *c.at_mut(idx) = *a.at(idx) + *b.at(idx);
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd_2d_f32(
    a: &GridNDDevice<f32, 2>,
    b: &GridNDDevice<f32, 2>,
    c: *mut GridNDDevice<f32, 2>,
) {
    let a: &GridND<'static, f32, 2> = a.into();
    let b: &GridND<'static, f32, 2> = b.into();
    // Safety: 'c' must point to a valid GridNDDevice<f32, 1> that is mutable.
    let c: &mut GridND<'static, f32, 2> = unsafe { &mut *c }.into();
    let (idx_x, idx_y) = thread::index_2d().as_usize_tuple();
    let dims = a.shape();
    assert!(dims == b.shape() && dims == c.shape());
    if idx_x < dims[1] && idx_y < dims[0] {
        *c.at_mut(idx_y).at_mut(idx_x) = *a.at(idx_y).at(idx_x) + *b.at(idx_y).at(idx_x);
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd_3d_f32(
    a: &GridNDDevice<f32, 3>,
    b: &GridNDDevice<f32, 3>,
    c: *mut GridNDDevice<f32, 3>,
) {
    let a: &GridND<'static, f32, 3> = a.into();
    let b: &GridND<'static, f32, 3> = b.into();
    // Safety: 'c' must point to a valid GridNDDevice<f32, 1> that is mutable.
    let c: &mut GridND<'static, f32, 3> = unsafe { &mut *c }.into();
    let (idx_x, idx_y, idx_z) = thread::index_3d().as_usize_tuple();
    let dims = a.shape();
    assert!(dims == b.shape() && dims == c.shape());
    if idx_x < dims[2] && idx_y < dims[1] && idx_z < dims[0] {
        *c.at_mut(idx_z).at_mut(idx_y).at_mut(idx_x) =
            *a.at(idx_z).at(idx_y).at(idx_x) + *b.at(idx_z).at(idx_y).at(idx_x);
    }
}
