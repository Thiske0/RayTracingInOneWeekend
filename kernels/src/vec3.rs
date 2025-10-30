use core::ops::{self, Index, Range};

use crate::random::RandomRange;
use approx::{AbsDiffEq, RelativeEq};
#[cfg(not(target_os = "cuda"))]
use cust::DeviceCopy;
use gpu_builder::DeviceCopyBuilder;

use crate::{color::Color, random::Random};

pub type Real = f32;

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(DeviceCopy, Copy, Debug))]
#[derive(PartialEq, Clone, DeviceCopyBuilder)]
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

    pub fn at_axis(&self, axis: &Axis) -> Real {
        match axis {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
        }
    }
}

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

impl Vec3 {
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    pub fn sample_square(rng: &mut Random) -> Vec3 {
        let x = rng.random_range(-0.5..0.5);
        let y = rng.random_range(-0.5..0.5);
        Vec3::new(x, y, 0.0)
    }

    pub fn random(interval: Range<Real>, rng: &mut Random) -> Self {
        Vec3::new(
            rng.random_range(interval.clone()),
            rng.random_range(interval.clone()),
            rng.random_range(interval),
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

    pub fn random_cosine_direction(rng: &mut Random) -> Vec3 {
        let r1: Real = rng.random_range(0.0..1.0);
        let r2: Real = rng.random_range(0.0..1.0);

        let phi = 2.0 * (core::f32::consts::PI as Real) * r1;
        let x = phi.cos() * r2.sqrt();
        let y = phi.sin() * r2.sqrt();
        let z = (1.0 - r2).sqrt();

        Vec3::new(x, y, z)
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

    pub fn rotate(&self, axis: &Axis, angle_rad: Real) -> Self {
        let cos_theta = angle_rad.cos();
        let sin_theta = angle_rad.sin();
        match axis {
            Axis::X => Vec3::new(
                self.x,
                self.y * cos_theta - self.z * sin_theta,
                self.y * sin_theta + self.z * cos_theta,
            ),
            Axis::Y => Vec3::new(
                self.x * cos_theta + self.z * sin_theta,
                self.y,
                -self.x * sin_theta + self.z * cos_theta,
            ),
            Axis::Z => Vec3::new(
                self.x * cos_theta - self.y * sin_theta,
                self.x * sin_theta + self.y * cos_theta,
                self.z,
            ),
        }
    }

    pub fn scale(&self, scale: &Vec3) -> Self {
        Vec3::new(self.x * scale.x, self.y * scale.y, self.z * scale.z)
    }

    pub fn scale_inverse(&self, scale: &Vec3) -> Self {
        Vec3::new(self.x / scale.x, self.y / scale.y, self.z / scale.z)
    }
}

#[cfg(test)]
mod rotation_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rotate_x() {
        let vec = Vec3::new(0.0, 1.0, 0.0);
        let rotated = vec.rotate(&Axis::X, 90.0_f32.to_radians());
        assert_relative_eq!(rotated, Vec3::new(0.0, 0.0, 1.0));
        let rotated = vec.rotate(&Axis::X, -90.0_f32.to_radians());
        assert_relative_eq!(rotated, Vec3::new(0.0, 0.0, -1.0));
    }
    #[test]
    fn test_rotate_y() {
        let vec = Vec3::new(0.0, 0.0, 1.0);
        let rotated = vec.rotate(&Axis::Y, 90.0_f32.to_radians());
        assert_relative_eq!(rotated, Vec3::new(1.0, 0.0, 0.0));
        let rotated = vec.rotate(&Axis::Y, -90.0_f32.to_radians());
        assert_relative_eq!(rotated, Vec3::new(-1.0, 0.0, 0.0));
    }
    #[test]
    fn test_rotate_z() {
        let vec = Vec3::new(1.0, 0.0, 0.0);
        let rotated = vec.rotate(&Axis::Z, 90.0_f32.to_radians());
        assert_relative_eq!(rotated, Vec3::new(0.0, 1.0, 0.0));
        let rotated = vec.rotate(&Axis::Z, -90.0_f32.to_radians());
        assert_relative_eq!(rotated, Vec3::new(0.0, -1.0, 0.0));
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

impl ops::MulAssign<Real> for Vec3 {
    fn mul_assign(&mut self, other: Real) {
        *self = &*self * other;
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

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Debug, Clone, Copy, DeviceCopy))]
#[derive(DeviceCopyBuilder)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn items() -> [Axis; 3] {
        [Axis::X, Axis::Y, Axis::Z]
    }
}

impl Index<&Axis> for &Vec3 {
    type Output = Real;

    fn index(&self, index: &Axis) -> &Real {
        match index {
            Axis::X => &self.x,
            Axis::Y => &self.y,
            Axis::Z => &self.z,
        }
    }
}

impl Index<Axis> for &Vec3 {
    type Output = Real;

    fn index(&self, index: Axis) -> &Real {
        self.index(&index)
    }
}

impl RelativeEq for Vec3 {
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Real) -> bool {
        self.x.relative_eq(&other.x, epsilon, max_relative)
            && self.y.relative_eq(&other.y, epsilon, max_relative)
            && self.z.relative_eq(&other.z, epsilon, max_relative)
    }

    fn default_max_relative() -> Self::Epsilon {
        Real::default_max_relative()
    }
}

impl AbsDiffEq for Vec3 {
    type Epsilon = Real;

    fn default_epsilon() -> Self::Epsilon {
        Real::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon)
            && self.y.abs_diff_eq(&other.y, epsilon)
            && self.z.abs_diff_eq(&other.z, epsilon)
    }
}
