use core::ops::Range;

#[cfg(target_os = "cuda")]
use crate::vec3::Real;

#[cfg(target_os = "cuda")]
use gpu_rand::{DefaultRand, GpuRand};
#[cfg(target_os = "cuda")]
use rand_core::RngCore;

#[cfg(target_os = "cuda")]
pub type Random = DefaultRand;

#[cfg(not(target_os = "cuda"))]
use rand::{Rng, rngs::ThreadRng};
#[cfg(not(target_os = "cuda"))]
pub type Random = ThreadRng;

pub trait RandomRange<T> {
    fn random_range(&mut self, interval: Range<T>) -> T;
}

#[cfg(not(target_os = "cuda"))]
impl<T: rand::distr::uniform::SampleUniform + std::cmp::PartialOrd> RandomRange<T> for Random {
    fn random_range(&mut self, interval: Range<T>) -> T {
        Rng::random_range(self, interval)
    }
}

#[cfg(target_os = "cuda")]
impl RandomRange<Real> for Random {
    fn random_range(&mut self, interval: Range<Real>) -> Real {
        self.uniform_f32() as Real * (interval.end - interval.start) + interval.start
    }
}

#[cfg(target_os = "cuda")]
impl RandomRange<usize> for Random {
    fn random_range(&mut self, interval: Range<usize>) -> usize {
        (self.next_u64() as usize % (interval.end - interval.start)) + interval.start
    }
}
