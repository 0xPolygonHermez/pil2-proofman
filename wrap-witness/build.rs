//! Compiles `src/zkin.cpp` against the system's nlohmann/json, the one `setup/final_snark_circom/`
//! compiles final.so against: the zkin it makes crosses into final.so's `getWitness` as a C++ object.

fn main() {
    println!("cargo:rerun-if-changed=src/zkin.cpp");
    cc::Build::new().cpp(true).std("c++17").warnings(true).extra_warnings(true).file("src/zkin.cpp").compile("zkin");
}
