cargo buld --features gpu --workspace
cargo run --features gpu --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square.so \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --output-dir examples/fibonacci-square/build/proofs \
     --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin -y -t 1
