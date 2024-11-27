use std::env;

fn main() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-cfg=feature=\"no_lib_link\"");
        return;
    }
    // Check if the "NO_LIB_LINK" feature is enabled
    if env::var("CARGO_FEATURE_NO_LIB_LINK").is_err() {

        let library_short_name = if cfg!(feature = "gpu") {
            "starksgpu"
        } else {
            "starks"
        };        
        let library_folder: String = if cfg!(feature = "gpu") { 
            "../../pil2-stark/lib-gpu".to_string()
        } else {
            "../../pil2-stark/lib".to_string()
        };
        let library_path = format!("{}/lib{}.a", library_folder, library_short_name);

        println!("Library folder: {}  and library path: {}", library_folder, library_path);
        println!("cargo:rustc-link-search=native={}", library_folder);
        println!("cargo:rustc-link-lib=static={}", library_short_name);
        if cfg!(feature="gpu"){
            // Add the CUDA library path
            let cuda_path = "/usr/local/cuda/lib64"; // Adjust this path if necessary
            println!("cargo:rustc-link-search=native={}", cuda_path);
            println!("cargo:rustc-link-lib=dylib=cudart"); // Link the CUDA runtime library

            // Specify the CUDA architecture
            println!("cargo:rustc-env=CUDA_ARCH=sm_75"); // Adjust the architecture as needed
        }

        // Trigger a rebuild if the library path changes
        let current_dir = env::current_dir().unwrap();
        println!("cargo:rerun-if-changed={}", current_dir.join(&library_path).display());

        // Add the library folder to the linker search path
        println!("cargo:rustc-link-search=native={}", current_dir.join(&library_folder).display());

        // Add additional common library search paths
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/");
        println!("cargo:rustc-link-search=native=/usr/lib/gcc/x86_64-linux-gnu/11/");

        // Add linker arguments
        println!("cargo:rustc-link-arg=-fopenmp");
        println!("cargo:rustc-link-arg=-MMD");
        println!("cargo:rustc-link-arg=-MP");
        println!("cargo:rustc-link-arg=-std=c++17");
        println!("cargo:rustc-link-arg=-Wall");
        println!("cargo:rustc-link-arg=-flarge-source-files");
        println!("cargo:rustc-link-arg=-Wno-unused-label");
        println!("cargo:rustc-link-arg=-rdynamic");
        println!("cargo:rustc-link-arg=-mavx2");

        // Link the static library specified by `library_short_name`
        println!("cargo:rustc-link-lib=static={}", library_short_name);

        // Link other required libraries
        for lib in &["sodium", "pthread", "gmp", "stdc++", "gmpxx", "crypto", "iomp5"] {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }
}
