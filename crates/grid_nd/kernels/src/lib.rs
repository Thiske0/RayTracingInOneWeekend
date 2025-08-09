#![feature(generic_const_exprs)]
#![feature(int_roundings)]
#![allow(incomplete_features)]

pub mod grid_nd;
mod test;

use cuda_std::glam::{UVec2, UVec3};
pub use grid_nd::{GridND, GridViewND, GridViewNDMut};

// Helper types for const generic constraints
pub enum Assert<const COND: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

trait TupleConversion2D {
    fn as_usize_tuple(self) -> (usize, usize);
}

impl TupleConversion2D for UVec2 {
    fn as_usize_tuple(self) -> (usize, usize) {
        (self.x as usize, self.y as usize)
    }
}

trait TupleConversion3D {
    fn as_usize_tuple(self) -> (usize, usize, usize);
}

impl TupleConversion3D for UVec3 {
    fn as_usize_tuple(self) -> (usize, usize, usize) {
        (self.x as usize, self.y as usize, self.z as usize)
    }
}
