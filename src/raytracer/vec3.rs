use std::{
    cell::UnsafeCell,
    ops::{self, Range},
};

use derive_more::{Add, AddAssign, Display, Mul, Neg, Sub, SubAssign};
use rand::Rng;

use crate::raytracer::color::Color;

pub type Real = f32;

#[derive(Debug, Clone, Copy, PartialEq, Add, AddAssign, Sub, SubAssign, Mul, Display, Neg)]
#[display("({}, {}, {})", x, y, z)]
pub struct Vec3 {
    pub x: Real,
    pub y: Real,
    pub z: Real,
}

thread_local! {
    static RNG: UnsafeCell<rand::rngs::ThreadRng> = UnsafeCell::new(rand::rng());
}

/// For clarity
pub type Point3 = Vec3;

impl Vec3 {
    pub fn new(x: Real, y: Real, z: Real) -> Self {
        Vec3 { x, y, z }
    }

    pub fn zero() -> Self {
        Vec3::new(0.0, 0.0, 0.0)
    }

    pub fn dot(self, other: Vec3) -> Real {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn length_squared(self) -> Real {
        self.dot(self)
    }

    pub fn length(self) -> Real {
        self.length_squared().sqrt()
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

    pub fn map(self, f: fn(Real) -> Real) -> Self {
        Vec3::new(f(self.x), f(self.y), f(self.z))
    }

    pub fn to_color(self) -> Color {
        Color::new(self.x, self.y, self.z)
    }

    pub fn near_zero(self) -> bool {
        let s = 1e-8;
        (self.x.abs() < s) && (self.y.abs() < s) && (self.z.abs() < s)
    }

    pub fn reflect(self, normal: Vec3) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }

    pub fn refract(self, normal: Vec3, etai_over_etat: Real) -> Self {
        let cos_theta = Real::min(-self.dot(normal), 1.0);
        let r_out_perp = (self + normal * cos_theta) * etai_over_etat;
        let r_out_parallel = normal * -Real::abs(1.0 - r_out_perp.length_squared()).sqrt();
        r_out_perp + r_out_parallel
    }
}

impl Vec3 {
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    pub fn sample_square() -> Vec3 {
        RNG.with(|rng| {
            // Safety: we only have one &mut to the RNG at a time.
            let rng = unsafe { &mut *rng.get() };
            let x = rng.random::<Real>() - 0.5;
            let y = rng.random::<Real>() - 0.5;
            Vec3::new(x, y, 0.0)
        })
    }

    pub fn random(interval: Range<Real>) -> Self {
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
            let length_squared = v.length_squared();
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

    pub fn random_in_unit_disk() -> Vec3 {
        loop {
            let mut v = Vec3::random(-1.0..1.0);
            v.z = 0.0; // Ensure it's in the disk
            let length_squared = v.length_squared();
            // Avoid division by zero by ensuring that the vector length is not too close to zero.
            if length_squared > 1e-15 && length_squared < 1.0 {
                return v / length_squared.sqrt();
            }
        }
    }
}

impl ops::Add<Real> for Vec3 {
    type Output = Self;

    fn add(self, scalar: Real) -> Self::Output {
        Vec3::new(self.x + scalar, self.y + scalar, self.z + scalar)
    }
}

impl ops::Div<Real> for Vec3 {
    type Output = Self;

    fn div(self, scalar: Real) -> Self::Output {
        Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}
