extern crate prost_build;

fn main() {
    let mut config = prost_build::Config::new();

    // Enable serde Serialize/Deserialize for all types and enforce camelCase
    config.type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)] #[serde(rename_all = \"camelCase\")]");

    // Enable experimental proto3 optional fields
    config.protoc_arg("--experimental_allow_proto3_optional");

    // Compile Protobuf definitions
    config.compile_protos(&["src/pilout.proto"], &["src/"]).unwrap();
}
