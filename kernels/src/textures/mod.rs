use crate::{
    color::Color,
    random::Random,
    textures::solid::Solid,
    vec3::{Point3, Real},
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait Texture {
    /// Returns the color of the texture at the given UV coordinates and point.
    fn color<'a>(&'a self, u: Real, v: Real, p: &Point3, rng: &mut Random) -> &'a Color;
}

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
#[enum_dispatch(Texture)]
pub enum TextureKind {
    Solid(Solid),
}

pub mod solid;
