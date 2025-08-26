use std::{any::Any, collections::HashMap, marker::PhantomData};

use thiserror::Error;

// Key that includes both TypeId and additional type information
#[derive(Hash, Eq, PartialEq, Clone)]
struct CacheKey {
    type_name: &'static str, // Or use TypeName from typetag crate
}

impl CacheKey {
    fn new<T>() -> Self {
        CacheKey {
            type_name: std::any::type_name::<T>(),
        }
    }
}

struct CacheEntry<'a> {
    value: *mut (),
    lifetime: PhantomData<&'a ()>,
}

impl<'a> CacheEntry<'a> {
    fn new<T>(value: T) -> Self {
        let value = Box::new(value);
        let value_ptr = Box::into_raw(value);
        CacheEntry {
            value: value_ptr as *mut (),
            lifetime: PhantomData,
        }
    }
    unsafe fn as_ref<T>(&self) -> &'a T {
        unsafe { (self.value as *const T).as_ref().unwrap() }
    }
    unsafe fn as_mut<T>(&mut self) -> &'a mut T {
        unsafe { (self.value as *mut T).as_mut().unwrap() }
    }
}

impl Drop for CacheEntry<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = Box::from_raw(self.value as *mut dyn Any);
        }
    }
}

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("item already exists in cache")]
    AlreadyExists,

    #[error("other error: {0}")]
    Other(String),
}

pub struct Cache<'a> {
    cache: HashMap<CacheKey, CacheEntry<'a>>,
}

impl<'a> Cache<'a> {
    pub fn new() -> Self {
        Cache {
            cache: HashMap::new(),
        }
    }

    pub fn get<T>(&self) -> Option<&'a T> {
        let key = CacheKey::new::<T>();
        self.cache
            .get(&key)
            .and_then(|entry| unsafe { Some(entry.as_ref()) })
    }

    pub fn get_mut<T>(&mut self) -> Option<&'a mut T> {
        let key = CacheKey::new::<T>();
        self.cache
            .get_mut(&key)
            .and_then(|entry| unsafe { Some(entry.as_mut()) })
    }

    pub fn insert<T>(&mut self, value: T) -> Result<(), CacheError> {
        let key = CacheKey::new::<T>();
        let entry = CacheEntry::new(value);
        if self.cache.insert(key, entry).is_none() {
            Ok(())
        } else {
            Err(CacheError::AlreadyExists)
        }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
