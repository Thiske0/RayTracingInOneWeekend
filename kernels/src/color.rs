use core::ops;

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
#[cfg(not(target_os = "cuda"))]
use image::Rgb;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

use crate::vec3::{Real, Vec3};

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
#[derive(PartialEq)]
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
}

#[cfg(not(target_os = "cuda"))]
impl Color {
    pub fn random() -> Self {
        Color(Vec3::random(0.0..1.0))
    }
}

#[cfg(target_os = "cuda")]
use gpu_rand::DefaultRand;

#[cfg(target_os = "cuda")]
impl Color {
    pub fn random(rng: &mut DefaultRand) -> Self {
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
