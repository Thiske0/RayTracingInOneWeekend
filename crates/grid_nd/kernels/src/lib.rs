#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod grid_nd;

pub use grid_nd::{GridND, GridViewND, GridViewNDMut};

#[cfg(target_os = "cuda")]
use cuda_std::prelude::*;

// Helper types for const generic constraints
pub enum Assert<const COND: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> {}
