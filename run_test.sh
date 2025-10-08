export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi)
export PROOFMAN_HOST_COMPILER_BIN=/usr/bin/g++-10
if [[ "$1" == "compile" ]]; then
    cargo build --release --features gpu
    if [ $? -ne 0 ]; then
        echo "Compilation failed! Exiting..."
        exit 1
    fi
fi
./target/release/proofman-cli prove  --witness-lib ./target/release/libfibonacci_square${PIL2_PROOFMAN_EXT}      --proving-key examples/fibonacci-square/build/provingKey/      --public-inputs examples/fibonacci-square/src/inputs.json      --output-dir examples/fibonacci-square/build/proofs      --custom-commits rom=examples/fibonacci-square/build/rom.bin -t 1 -y 

#compute-sanitizer --tool racecheck ./target/debug/proofman-cli prove  --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT}      --proving-key examples/fibonacci-square/build/provingKey/      --public-inputs examples/fibonacci-square/src/inputs.json      --output-dir examples/fibonacci-square/build/proofs      --custom-commits rom=examples/fibonacci-square/build/rom.bin -y > k