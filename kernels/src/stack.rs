use core::mem;
use core::panic;

const STACK_SIZE: usize = 64;

pub struct Stack {
    data: [u8; STACK_SIZE],
    top: usize,
}

impl Stack {
    pub fn new() -> Self {
        Self {
            data: [0; STACK_SIZE],
            top: 0,
        }
    }

    pub fn push<T>(&mut self, value: T) {
        let size = mem::size_of::<T>();
        let start = self.top;
        let end = start + size;
        if end > STACK_SIZE {
            panic!("Stack overflow");
        }
        let ptr = self.data[start..end].as_mut_ptr() as *mut T;
        unsafe {
            *ptr = value;
        }
        self.top = end;
    }

    pub unsafe fn pop<T>(&mut self) -> T {
        let size = mem::size_of::<T>();
        let start = self.top - size;
        if self.top < size {
            panic!("Stack underflow");
        }
        let ptr = self.data[start..self.top].as_ptr() as *const T;
        self.top = start;
        unsafe { ptr.read() }
    }
}
