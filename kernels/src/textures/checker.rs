use crate::{
    color::Color,
    random::Random,
    textures::Texture,
    vec3::{Point3, Real},
};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
pub struct CheckerTexture {
    odd: Color,
    even: Color,
    inv_scale: Real,
}

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

impl CheckerTexture {
    pub fn new(odd: Color, even: Color, scale: Real) -> Self {
        CheckerTexture {
            odd,
            even,
            inv_scale: 1.0 / scale,
        }
    }
}

impl Texture for CheckerTexture {
    fn color<'a>(&'a self, _u: Real, _v: Real, p: &Point3, _rng: &mut Random) -> &'a Color {
        let x_integer = (self.inv_scale * p.x).floor() as i32;
        let y_integer = (self.inv_scale * p.y).floor() as i32;
        let z_integer = (self.inv_scale * p.z).floor() as i32;
        if (x_integer + y_integer + z_integer) % 2 == 0 {
            &self.odd
        } else {
            &self.even
        }
    }
}
