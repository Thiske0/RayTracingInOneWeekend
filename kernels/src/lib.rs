#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#[cfg(target_os = "cuda")]
use cuda_std::prelude::*;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

use crate::vec3::{Point3, Real, Vec3};

pub mod color;
pub mod hitable;
pub mod hitable_list;
pub mod materials;
pub mod ray;
pub mod sphere;
pub mod vec3;

#[cfg_attr(not(target_os = "cuda"), derive(DeviceCopy))]
#[derive(Clone, Copy)]
pub struct ImageRenderOptions {
    pub samples_per_pixel: usize,
    pub origin: Point3,
    pub max_depth: usize,

    pub defocus_angle: Real,
    pub defocus_disk_u: Vec3,
    pub defocus_disk_v: Vec3,

    pub pixel00_loc: Point3,
    pub pixel_delta_u: Vec3,
    pub pixel_delta_v: Vec3,
}
