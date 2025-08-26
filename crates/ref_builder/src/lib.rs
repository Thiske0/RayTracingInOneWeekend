#![feature(iterator_try_collect)]

use core::ops::Deref;
use gpu_builder::{BuildResultType, Builder, DeviceImpl};

#[cfg(not(target_os = "cuda"))]
mod implementation;

#[repr(C)]
pub struct RefBuilder<'a, T: Builder<'a>> {
    reference: *const T,
    _marker: core::marker::PhantomData<&'a T>,
}

#[repr(C)]
#[derive(DeviceImpl)]
#[builder(RefBuilder)]
#[use_lifetime("'a")]
pub struct RefBuilderDevice<T: BuildResultType> {
    reference: *const T,
}

impl<'a, T: Builder<'a>> RefBuilder<'a, T> {
    pub fn as_ref(&self) -> &'a T {
        unsafe { &*self.reference }
    }
}

impl<'a, T: Builder<'a>> Deref for RefBuilder<'a, T> {
    type Target = T;

    fn deref(&self) -> &'a T {
        self.as_ref()
    }
}

#[repr(C)]
pub struct SliceBuilder<'a, T: Builder<'a>> {
    reference: *const T,
    size: usize,
    _marker: core::marker::PhantomData<&'a [T]>,
}

#[repr(C)]
#[derive(DeviceImpl)]
#[builder(SliceBuilder)]
#[use_lifetime("'a")]
pub struct SliceBuilderDevice<T: BuildResultType> {
    reference: *const T,
    size: usize,
}

impl<'a, T: Builder<'a>> SliceBuilder<'a, T> {
    pub fn as_slice(&self) -> &'a [T] {
        unsafe { core::slice::from_raw_parts(self.reference, self.size) }
    }
}

impl<'a, T: Builder<'a>> Deref for SliceBuilder<'a, T> {
    type Target = [T];

    fn deref(&self) -> &'a [T] {
        self.as_slice()
    }
}
