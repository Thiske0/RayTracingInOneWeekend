pub mod raytracer;

use std::error::Error;

use indicatif::ProgressBar;

// Simplify error handling with type-erased errors
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn print_wrapper(func: impl Fn() -> (), progress_bar: Option<ProgressBar>) {
    if let Some(progress) = progress_bar {
        // Suspend the progress bar while running the function
        progress.suspend(func);
    } else {
        // Otherwise, just run the function
        func();
    }
}
