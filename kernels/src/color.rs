use core::ops;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
use gpu_builder::DeviceCopyBuilder;
#[cfg(not(target_os = "cuda"))]
use image::Rgb;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

use crate::{
    random::Random,
    vec3::{Real, Vec3},
};

#[cfg_attr(not(target_os = "cuda"), derive(Copy, DeviceCopy, Debug))]
#[repr(C)]
#[derive(DeviceCopyBuilder, PartialEq, Clone)]
pub struct Color(pub(crate) Vec3);

impl Color {
    pub fn new(r: Real, g: Real, b: Real) -> Self {
        Color(Vec3::new(r, g, b))
    }

    fn linear_to_gamma(&self) -> Color {
        Color(self.0.map(|c| if c > 0.0 { c.sqrt() } else { 0.0 }))
    }

    fn to_rgb(&self) -> (i32, i32, i32) {
        let v = (self.linear_to_gamma() * 255.999)
            .0
            .map(|c| c.clamp(0.0, 255.0));
        (v.x as i32, v.y as i32, v.z as i32)
    }

    pub fn lerp(&self, other: &Color, t: Real) -> Self {
        self + (other - self) * t
    }

    pub fn black() -> Self {
        Color(Vec3::zero())
    }

    pub fn white() -> Self {
        Color(Vec3::new(1.0, 1.0, 1.0))
    }

    pub fn random(rng: &mut Random) -> Self {
        Color(Vec3::random(0.0..1.0, rng))
    }
}

#[cfg(not(target_os = "cuda"))]
impl Into<Rgb<u8>> for Color {
    fn into(self) -> Rgb<u8> {
        let (r, g, b) = self.to_rgb();
        Rgb([r as u8, g as u8, b as u8])
    }
}

impl ops::Mul<&Color> for &Color {
    type Output = Color;

    fn mul(self, other: &Color) -> Self::Output {
        Color::new(
            self.0.x * other.0.x,
            self.0.y * other.0.y,
            self.0.z * other.0.z,
        )
    }
}

impl ops::Mul<Color> for &Color {
    type Output = Color;

    fn mul(self, other: Color) -> Self::Output {
        self * &other
    }
}

impl ops::Mul<&Color> for Color {
    type Output = Color;

    fn mul(self, other: &Color) -> Self::Output {
        &self * other
    }
}

impl ops::Mul<Color> for Color {
    type Output = Color;

    fn mul(self, other: Color) -> Self::Output {
        &self * &other
    }
}

impl ops::Add<&Color> for &Color {
    type Output = Color;

    fn add(self, other: &Color) -> Self::Output {
        Color(&self.0 + &other.0)
    }
}

impl ops::Add<Color> for &Color {
    type Output = Color;

    fn add(self, other: Color) -> Self::Output {
        Color(&self.0 + &other.0)
    }
}

impl ops::Add<&Color> for Color {
    type Output = Color;

    fn add(self, other: &Color) -> Self::Output {
        Color(&self.0 + &other.0)
    }
}

impl ops::Add<Color> for Color {
    type Output = Color;

    fn add(self, other: Color) -> Self::Output {
        Color(&self.0 + &other.0)
    }
}

impl ops::Add<Real> for &Color {
    type Output = Color;

    fn add(self, scalar: Real) -> Self::Output {
        Color(&self.0 + scalar)
    }
}

impl ops::Add<Real> for Color {
    type Output = Color;

    fn add(self, scalar: Real) -> Self::Output {
        Color(&self.0 + scalar)
    }
}

impl ops::AddAssign<&Color> for Color {
    fn add_assign(&mut self, other: &Color) {
        self.0 += &other.0;
    }
}

impl ops::AddAssign<Color> for Color {
    fn add_assign(&mut self, other: Color) {
        self.0 += &other.0;
    }
}

impl ops::Sub<&Color> for &Color {
    type Output = Color;

    fn sub(self, other: &Color) -> Self::Output {
        Color(&self.0 - &other.0)
    }
}

impl ops::Sub<Color> for &Color {
    type Output = Color;

    fn sub(self, other: Color) -> Self::Output {
        Color(&self.0 - &other.0)
    }
}

impl ops::Sub<&Color> for Color {
    type Output = Color;

    fn sub(self, other: &Color) -> Self::Output {
        Color(&self.0 - &other.0)
    }
}

impl ops::Sub<Color> for Color {
    type Output = Color;

    fn sub(self, other: Color) -> Self::Output {
        Color(&self.0 - &other.0)
    }
}

impl ops::Sub<Real> for &Color {
    type Output = Color;

    fn sub(self, scalar: Real) -> Self::Output {
        Color(&self.0 - scalar)
    }
}

impl ops::Sub<Real> for Color {
    type Output = Color;

    fn sub(self, scalar: Real) -> Self::Output {
        Color(&self.0 - scalar)
    }
}

impl ops::SubAssign<&Color> for Color {
    fn sub_assign(&mut self, other: &Color) {
        self.0 -= &other.0;
    }
}

impl ops::SubAssign<Color> for Color {
    fn sub_assign(&mut self, other: Color) {
        self.0 -= &other.0;
    }
}

impl ops::Mul<Real> for &Color {
    type Output = Color;

    fn mul(self, scalar: Real) -> Self::Output {
        Color(&self.0 * scalar)
    }
}

impl ops::Mul<Real> for Color {
    type Output = Color;

    fn mul(self, scalar: Real) -> Self::Output {
        Color(&self.0 * scalar)
    }
}

impl ops::Div<Real> for &Color {
    type Output = Color;

    fn div(self, scalar: Real) -> Self::Output {
        Color(&self.0 / scalar)
    }
}

impl ops::Div<Real> for Color {
    type Output = Color;

    fn div(self, scalar: Real) -> Self::Output {
        Color(&self.0 / scalar)
    }
}

#[cfg(not(target_os = "cuda"))]
use core::str::FromStr;

#[cfg(not(target_os = "cuda"))]
impl FromStr for Color {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err("Expected three comma-separated values".to_string());
        }
        let r = parts[0]
            .parse()
            .map_err(|_| "Invalid r value".to_string())?;
        let g = parts[1]
            .parse()
            .map_err(|_| "Invalid g value".to_string())?;
        let b = parts[2]
            .parse()
            .map_err(|_| "Invalid b value".to_string())?;
        Ok(Color::new(r, g, b))
    }
}
