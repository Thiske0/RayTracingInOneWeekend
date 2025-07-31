use std::{
    cell::UnsafeCell,
    ops::{self, Range},
};

use derive_more::{Add, AddAssign, Display, Mul, Neg, Sub, SubAssign};
use rand::Rng;

use crate::raytracer::color::Color;

#[derive(Debug, Clone, Copy, PartialEq, Add, AddAssign, Sub, SubAssign, Mul, Display, Neg)]
#[display("({}, {}, {})", x, y, z)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

thread_local! {
    static RNG: UnsafeCell<rand::rngs::ThreadRng> = UnsafeCell::new(rand::rng());
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

impl Vec3 {
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    pub fn sample_square() -> Vec3 {
        RNG.with(|rng| {
            // Safety: we only have one &mut to the RNG at a time.
            let rng = unsafe { &mut *rng.get() };
            let x = rng.random::<f32>() - 0.5;
            let y = rng.random::<f32>() - 0.5;
            Vec3::new(x, y, 0.0)
        })
    }

    fn random(interval: Range<f32>) -> Self {
        RNG.with(|rng| {
            // Safety: we only have one &mut to the RNG at a time.
            let rng = unsafe { &mut *rng.get() };
            Vec3::new(
                rng.random_range(interval.clone()),
                rng.random_range(interval.clone()),
                rng.random_range(interval),
            )
        })
    }

    pub fn random_unit() -> Self {
        loop {
            let v = Vec3::random(-1.0..1.0);
            let length_squared = v.dot(v);
            // Avoid division by zero by ensuring that the vector length is not too close to zero.
            if length_squared > 1e-15 && length_squared < 1.0 {
                return v / length_squared.sqrt();
            }
        }
    }

    pub fn random_on_hemisphere(normal: Vec3) -> Vec3 {
        let on_unit_sphere = Vec3::random_unit();
        if on_unit_sphere.dot(normal) > 0.0 {
            // In the same hemisphere as the normal
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
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
