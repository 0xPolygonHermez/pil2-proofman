use std::path::PathBuf;

use stark_recurser::{main_gen_recursive1_vadcop, main_gen_compressor_vadcop};

fn main() {
    let proving_key_path = PathBuf::from("examples/fibonacci-square/build/provingKey/");
    let air_id = 1;
    let airgroup_id = 0;

    // Call the function that generates and writes the compressed template
    if let Err(e) = main_gen_compressor_vadcop(proving_key_path.clone(), airgroup_id, air_id) {
        eprintln!("Error: {}", e);
    }

    if let Err(e) = main_gen_recursive1_vadcop(proving_key_path.clone(), 0, 0, false) {
        eprintln!("Error: {}", e);
    }
}
