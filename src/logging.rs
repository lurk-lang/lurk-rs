// platform independent logging module and macros
#[cfg(not(target_arch = "wasm32"))]
use std::io::{self, Write};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = "log")]
    pub fn js_log(s: &str);
}

/// Logs to js console when compiled with debug
#[cfg(target_arch = "wasm32")]
#[cfg(debug_assertions)]
pub fn debug(s: &str) {
    js_log(&["[DEBUG]", s].join(" "));
}

#[cfg(not(debug_assertions))]
pub fn debug(_: &str) {
    // ignore
}

/// Logs to js console
#[cfg(target_arch = "wasm32")]
pub fn log(s: &str) {
    js_log(&["[INFO]", s].join(" "));
}

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => ($crate::logging::log(&format!($($arg)*)));
}

// Not WASM
/// Logs when compiled with debug
#[cfg(debug_assertions)]
#[cfg(not(target_arch = "wasm32"))]
pub fn debug(s: &str) {
    io::stdout()
        .write_all(["[DEBUG]", s, "\n"].join(" ").as_bytes())
        .unwrap();
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => ($crate::logging::debug(&format!($($arg)*)));
}

/// Logs info to std out
#[cfg(not(target_arch = "wasm32"))]
pub fn log(s: &str) {
    io::stdout()
        .write_all(["[INFO]", s, "\n"].join(" ").as_bytes())
        .unwrap();
}
