/// Module for wasm-bindgen specific handling and endpoints.
use crate::{
    eval::{empty_sym_env, Evaluator},
    store::{ContTag, Pointer, Store},
    writer::Write,
};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

use blstrs::Scalar as Fr;
use serde_json::json;

/// Run a lurk snippet
#[wasm_bindgen(catch)]
pub fn run_lurk(source: JsValue) -> Result<JsValue, JsValue> {
    let limit = 100_000_000;

    let expression = source
        .as_string()
        .ok_or_else(|| "input source must be a string")?;

    let mut store = Store::<Fr>::default();
    let mut context: HashMap<&str, String> = HashMap::new();

    context.insert("expression", expression.clone());

    if let Some(expr) = store.read(&expression) {
        let (output, iterations, _ptrs) =
            Evaluator::new(expr, empty_sym_env(&store), &mut store, limit).eval();

        let iterations_str = iterations.to_string();
        context.insert("iterations", iterations_str);
        let result_str = match output.cont.tag() {
            ContTag::Outermost | ContTag::Terminal => {
                let result = store.fetch(&output.expr).clone().unwrap();
                result.fmt_to_string(&store)
            }
            ContTag::Error => "ERROR!".to_string(),
            _ => format!("Computation incomplete after limit: {}", limit),
        };

        context.insert("result", result_str);
    } else {
        let error = format!("Syntax Error: {}", &expression);
        context.insert("result", error);
    }
    let json = json!(&context);
    Ok(json.to_string().into())
}
