use crate::{GridND, TupleConversion2D, TupleConversion3D};
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd_1d_f32(a: &GridND<f32, 1>, b: &GridND<f32, 1>, c: *mut GridND<f32, 1>) {
    // Safety: 'c' must point to a valid GridND<f32, 1> that is mutable.
    let c = unsafe { &mut *c };
    let idx = thread::index_1d() as usize;
    let dims = a.shape();
    assert!(dims == b.shape() && dims == c.shape());
    if idx < dims[0] {
        *c.at_mut(idx) = *a.at(idx) + *b.at(idx);
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd_2d_f32(a: &GridND<f32, 2>, b: &GridND<f32, 2>, c: *mut GridND<f32, 2>) {
    // Safety: 'c' must point to a valid GridND<f32, 2> that is mutable.
    let c = unsafe { &mut *c };
    let (idx_x, idx_y) = thread::index_2d().as_usize_tuple();
    let dims = a.shape();
    assert!(dims == b.shape() && dims == c.shape());
    if idx_x < dims[0] && idx_y < dims[1] {
        *c.at_mut(idx_x).at_mut(idx_y) = *a.at(idx_x).at(idx_y) + *b.at(idx_x).at(idx_y);
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd_3d_f32(a: &GridND<f32, 3>, b: &GridND<f32, 3>, c: *mut GridND<f32, 3>) {
    // Safety: 'c' must point to a valid GridND<f32, 3> that is mutable.
    let c = unsafe { &mut *c };
    let (idx_x, idx_y, idx_z) = thread::index_3d().as_usize_tuple();
    let dims = a.shape();
    assert!(dims == b.shape() && dims == c.shape());
    if idx_x < dims[0] && idx_y < dims[1] && idx_z < dims[2] {
        *c.at_mut(idx_x).at_mut(idx_y).at_mut(idx_z) =
            *a.at(idx_x).at(idx_y).at(idx_z) + *b.at(idx_x).at(idx_y).at(idx_z);
    }
}
