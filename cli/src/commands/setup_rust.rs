// use clap::Parser;
// use colored::Colorize;
// use deno_core::{extension, op2, JsRuntime, OpState, RuntimeOptions};
// use serde::{Deserialize, Serialize};

// // Generic state containers using JSON
// #[derive(Default)]
// struct JsInput(serde_json::Value);

// #[derive(Default)]
// struct JsOutput(Option<serde_json::Value>);

// #[op2]
// #[serde]
// fn op_get_input(state: &OpState) -> serde_json::Value {
//     state.borrow::<JsInput>().0.clone()
// }

// #[op2]
// fn op_store_result(state: &mut OpState, #[serde] value: serde_json::Value) {
//     state.borrow_mut::<JsOutput>().0 = Some(value);
// }

// extension!(rust_setup_ext, ops = [op_get_input, op_store_result],);

// #[derive(Parser)]
// #[command(version, about = "Run Rust setup with JS runtime", long_about = None)]
// #[command(propagate_version = true)]
// pub struct SetupRustCmd {}

// impl SetupRustCmd {
//     pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         println!("{} RustSetup", format!("{: >12}", "Command").bright_green().bold());
//         println!();

//         let rt = tokio::runtime::Runtime::new()?;
//         rt.block_on(self.run_async())?;

//         Ok(())
//     }

//     async fn run_async(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         // Any serializable input works now
//         #[derive(Serialize)]
//         struct MyInput {
//             value: f64,
//             name: String,
//         }

//         let input = MyInput { value: 10.5, name: "test".to_string() };

//         let mut runtime =
//             JsRuntime::new(RuntimeOptions { extensions: vec![rust_setup_ext::init()], ..Default::default() });

//         // Put input/output state into the runtime
//         {
//             let op_state = runtime.op_state();
//             let mut state = op_state.borrow_mut();
//             state.put(JsInput(serde_json::to_value(&input)?));
//             state.put(JsOutput::default());
//         }

//         // Load JS code from file (embedded at compile time)
//         let js_code = include_str!("../../assets/scripts/double.js");

//         // Execute script that calls our op with the result
//         let promise = runtime.execute_script("<init>", js_code)?;

//         // Resolve the promise
//         #[allow(deprecated)]
//         runtime.resolve_value(promise).await?;

//         // Get the result from OpState - can deserialize to any type
//         #[derive(Deserialize, Debug)]
//         #[allow(dead_code)]
//         struct MyOutput {
//             doubled: f64,
//             message: String,
//         }

//         let result: MyOutput = {
//             let op_state = runtime.op_state();
//             let state = op_state.borrow();
//             let value = state.borrow::<JsOutput>().0.clone().unwrap_or(serde_json::Value::Null);
//             serde_json::from_value(value)?
//         };
//         println!("Result = {:?}", result);

//         Ok(())
//     }
// }