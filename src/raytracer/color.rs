use core::fmt;

use derive_more::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use crate::raytracer::vec3::{Real, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Add, AddAssign, Sub, SubAssign, Mul, Div)]
pub struct Color(Vec3);

impl Color {
    pub fn new(r: Real, g: Real, b: Real) -> Self {
        Color(Vec3::new(r, g, b))
    }

    fn linear_to_gamma(self) -> Color {
        Color(self.0.map(|c| if c > 0.0 { c.sqrt() } else { 0.0 }))
    }

    pub fn to_rgb(self) -> (i32, i32, i32) {
        let v = (self.linear_to_gamma() * 255.999)
            .0
            .map(|c| c.clamp(0.0, 255.0));
        (v.x as i32, v.y as i32, v.z as i32)
    }

    pub fn lerp(self, other: Color, t: Real) -> Self {
        self + (other - self) * t
    }

    pub fn black() -> Self {
        Color(Vec3::zero())
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, g, b) = self.to_rgb();
        write!(f, "{} {} {}", r, g, b)
    }
}
