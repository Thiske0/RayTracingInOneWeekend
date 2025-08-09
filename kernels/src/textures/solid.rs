use crate::{
    color::Color,
    random::Random,
    textures::Texture,
    vec3::{Point3, Real},
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct Solid {
    color: Color,
}

impl Solid {
    pub fn new(color: Color) -> Self {
        Solid { color }
    }
}

impl Texture for Solid {
    fn color<'a>(&'a self, _u: Real, _v: Real, _p: &Point3, _rng: &mut Random) -> &'a Color {
        &self.color
    }
}
