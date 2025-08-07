use crate::vec3::Real;
use core::ops::Range;

#[cfg(target_os = "cuda")]
use gpu_rand::{DefaultRand, GpuRand};

#[cfg(target_os = "cuda")]
pub type Random = DefaultRand;

#[cfg(not(target_os = "cuda"))]
use rand::{Rng, rngs::ThreadRng};
#[cfg(not(target_os = "cuda"))]
pub type Random = ThreadRng;

#[cfg(not(target_os = "cuda"))]
pub fn random_single(interval: Range<Real>, rng: &mut Random) -> Real {
    rng.random_range(interval)
}

#[cfg(target_os = "cuda")]
pub fn random_single(interval: Range<Real>, rng: &mut Random) -> Real {
    rng.uniform_f32() as Real * (interval.end - interval.start) + interval.start
}
