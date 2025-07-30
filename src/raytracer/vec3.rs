use std::ops;

use derive_more::{Add, AddAssign, Display, Mul, Sub, SubAssign};

use crate::raytracer::color::Color;

#[derive(Debug, Clone, Copy, PartialEq, Add, AddAssign, Sub, SubAssign, Mul, Display)]
#[display("({}, {}, {})", x, y, z)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// For clarity
pub type Point3 = Vec3;

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    pub fn zero() -> Self {
        Vec3::new(0.0, 0.0, 0.0)
    }

    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        Vec3::new(self.x / len, self.y / len, self.z / len)
    }

    pub fn cross(self, other: Vec3) -> Self {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn map(self, f: fn(f32) -> f32) -> Self {
        Vec3::new(f(self.x), f(self.y), f(self.z))
    }

    pub fn to_color(self) -> Color {
        Color::new(self.x, self.y, self.z)
    }
}

impl ops::Add<f32> for Vec3 {
    type Output = Self;

    fn add(self, scalar: f32) -> Self::Output {
        Vec3::new(self.x + scalar, self.y + scalar, self.z + scalar)
    }
}

impl ops::Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, scalar: f32) -> Self::Output {
        Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}
