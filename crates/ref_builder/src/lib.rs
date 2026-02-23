#![feature(iterator_try_collect)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use core::ops::Deref;
use gpu_builder::{derive_device_struct, Builder};

#[cfg(not(target_os = "cuda"))]
mod implementation;

#[derive_device_struct]
#[repr(C)]
pub struct RefBuilder<'a, T: Builder<'a>> {
    reference: *const T,
    #[no_copy]
    _marker: core::marker::PhantomData<&'a T>,
    #[host_only]
    owned: bool,
}

#[cfg(not(target_os = "cuda"))]
impl<'a, T: Builder<'a>> Clone for RefBuilder<'a, T> {
    fn clone(&self) -> Self {
        if self.owned {
            #[cfg(debug_assertions)]
            println!("Cloning an owned RefBuilder is unsafe, make sure to not drop the original before the clone.");
        }
        RefBuilder {
            reference: self.reference,
            _marker: core::marker::PhantomData,
            owned: false,
        }
    }
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
#[derive_device_struct]
pub struct SliceBuilder<'a, T: Builder<'a>> {
    reference: *const T,
    size: usize,
    #[no_copy]
    _marker: core::marker::PhantomData<&'a [T]>,
    #[host_only]
    owned: bool,
}

#[cfg(not(target_os = "cuda"))]
impl<'a, T: Builder<'a>> Clone for SliceBuilder<'a, T> {
    fn clone(&self) -> Self {
        if self.owned {
            #[cfg(debug_assertions)]
            println!("Cloning an owned SliceBuilder is unsafe, make sure to not drop the original before the clone.");
        }
        SliceBuilder {
            reference: self.reference,
            size: self.size,
            _marker: core::marker::PhantomData,
            owned: false,
        }
    }
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
