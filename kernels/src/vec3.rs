use core::ops::{self, Range};

#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;

use crate::{
    color::Color,
    random::{Random, random_single},
};

pub type Real = f32;

#[cfg_attr(not(target_os = "cuda"), derive(DeviceCopy, Clone, Copy, Debug))]
#[derive(PartialEq)]
pub struct Vec3 {
    pub x: Real,
    pub y: Real,
    pub z: Real,
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

    pub fn dot(&self, other: &Vec3) -> Real {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn length_squared(&self) -> Real {
        self.dot(self)
    }

    pub fn length(&self) -> Real {
        self.length_squared().sqrt()
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        Vec3::new(self.x / len, self.y / len, self.z / len)
    }

    pub fn cross(&self, other: &Vec3) -> Self {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn map(&self, f: fn(Real) -> Real) -> Self {
        Vec3::new(f(self.x), f(self.y), f(self.z))
    }

    pub fn to_color(self) -> Color {
        Color::new(self.x, self.y, self.z)
    }

    pub fn near_zero(&self) -> bool {
        let s = 1e-8;
        (self.x.abs() < s) && (self.y.abs() < s) && (self.z.abs() < s)
    }

    pub fn reflect(&self, normal: &Vec3) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }

    pub fn refract(&self, normal: &Vec3, etai_over_etat: Real) -> Self {
        let cos_theta = Real::min(-self.dot(normal), 1.0);
        let r_out_perp = (self + normal * cos_theta) * etai_over_etat;
        let r_out_parallel = normal * -Real::abs(1.0 - r_out_perp.length_squared()).sqrt();
        r_out_perp + r_out_parallel
    }
}

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

impl Vec3 {
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    pub fn sample_square(rng: &mut Random) -> Vec3 {
        let x = random_single(-0.5..0.5, rng);
        let y = random_single(-0.5..0.5, rng);
        Vec3::new(x, y, 0.0)
    }

    pub fn random(interval: Range<Real>, rng: &mut Random) -> Self {
        Vec3::new(
            random_single(interval.clone(), rng),
            random_single(interval.clone(), rng),
            random_single(interval, rng),
        )
    }

    pub fn random_unit(rng: &mut Random) -> Self {
        loop {
            let v = Vec3::random(-1.0..1.0, rng);
            let length_squared = v.length_squared();
            // Avoid division by zero by ensuring that the vector length is not too close to zero.
            if length_squared > 1e-15 && length_squared < 1.0 {
                return v / length_squared.sqrt();
            }
        }
    }

    pub fn random_on_hemisphere(normal: Vec3, rng: &mut Random) -> Vec3 {
        let on_unit_sphere = Vec3::random_unit(rng);
        if on_unit_sphere.dot(&normal) > 0.0 {
            // In the same hemisphere as the normal
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }

    pub fn random_in_unit_disk(rng: &mut Random) -> Vec3 {
        loop {
            let mut v = Vec3::random(-1.0..1.0, rng);
            v.z = 0.0; // Ensure it's in the disk
            let length_squared = v.length_squared();
            // Avoid division by zero by ensuring that the vector length is not too close to zero.
            if length_squared > 1e-15 && length_squared < 1.0 {
                return v / length_squared.sqrt();
            }
        }
    }
}

impl ops::Add<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn add(self, other: &Vec3) -> Self::Output {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl ops::Add<&Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, other: &Vec3) -> Self::Output {
        &self + other
    }
}

impl ops::Add<Vec3> for &Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Self::Output {
        self + &other
    }
}

impl ops::Add<Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Self::Output {
        &self + &other
    }
}

impl ops::Add<Real> for &Vec3 {
    type Output = Vec3;

    fn add(self, scalar: Real) -> Self::Output {
        Vec3::new(self.x + scalar, self.y + scalar, self.z + scalar)
    }
}

impl ops::Add<Real> for Vec3 {
    type Output = Vec3;

    fn add(self, scalar: Real) -> Self::Output {
        &self + scalar
    }
}

impl ops::AddAssign<&Vec3> for Vec3 {
    fn add_assign(&mut self, other: &Vec3) {
        *self = &*self + other;
    }
}

impl ops::AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        *self += &other;
    }
}

impl ops::Sub<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, other: &Vec3) -> Self::Output {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl ops::Sub<Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Self::Output {
        self - &other
    }
}

impl ops::Sub<&Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, other: &Vec3) -> Self::Output {
        &self - other
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Self::Output {
        &self - &other
    }
}

impl ops::Sub<Real> for &Vec3 {
    type Output = Vec3;

    fn sub(self, scalar: Real) -> Self::Output {
        Vec3::new(self.x - scalar, self.y - scalar, self.z - scalar)
    }
}

impl ops::Sub<Real> for Vec3 {
    type Output = Vec3;

    fn sub(self, scalar: Real) -> Self::Output {
        &self - scalar
    }
}

impl ops::SubAssign<&Vec3> for Vec3 {
    fn sub_assign(&mut self, other: &Vec3) {
        *self = &*self - other;
    }
}

impl ops::SubAssign<Vec3> for Vec3 {
    fn sub_assign(&mut self, other: Vec3) {
        *self -= &other;
    }
}

impl ops::Mul<Real> for &Vec3 {
    type Output = Vec3;

    fn mul(self, scalar: Real) -> Self::Output {
        Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl ops::Mul<Real> for Vec3 {
    type Output = Vec3;

    fn mul(self, scalar: Real) -> Self::Output {
        &self * scalar
    }
}

impl ops::Div<Real> for &Vec3 {
    type Output = Vec3;

    fn div(self, scalar: Real) -> Self::Output {
        Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl ops::Div<Real> for Vec3 {
    type Output = Vec3;

    fn div(self, scalar: Real) -> Self::Output {
        &self / scalar
    }
}

impl ops::Neg for &Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        -&self
    }
}

#[cfg(not(target_os = "cuda"))]
use core::str::FromStr;

#[cfg(not(target_os = "cuda"))]
impl FromStr for Vec3 {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err("Expected three comma-separated values".to_string());
        }
        let x = parts[0]
            .parse()
            .map_err(|_| "Invalid x value".to_string())?;
        let y = parts[1]
            .parse()
            .map_err(|_| "Invalid y value".to_string())?;
        let z = parts[2]
            .parse()
            .map_err(|_| "Invalid z value".to_string())?;
        Ok(Vec3::new(x, y, z))
    }
}
