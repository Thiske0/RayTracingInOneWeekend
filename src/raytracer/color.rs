use core::fmt;

use derive_more::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use crate::raytracer::vec3::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Add, AddAssign, Sub, SubAssign, Mul, Div)]
pub struct Color(Vec3);

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Color(Vec3::new(r, g, b))
    }

    pub fn to_rgb(self) -> (i32, i32, i32) {
        let v = (self * 255.999).0.map(|c| c.clamp(0.0, 255.0));
        (v.x as i32, v.y as i32, v.z as i32)
    }

    pub fn lerp(self, other: Color, t: f32) -> Self {
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
