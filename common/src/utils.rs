//! System / runtime helpers: byte formatting, memory snapshot, rayon pool sizing, GPU init.

use crate::{ProofmanError, ProofmanResult};
use proofman_starks_lib_c::{get_num_gpus_c, init_gpu_setup_c, set_gpu_mode_c, GOLDILOCKS_MERKLE_TREE_ARITY};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::env;
use sysinfo::System;

pub fn format_bytes(mut num_bytes: f64) -> String {
    let units = ["Bytes", "KB", "MB", "GB"];
    let mut unit_index = 0;

    while num_bytes >= 0.01 && unit_index < units.len() - 1 {
        if num_bytes < 1024.0 {
            break;
        }
        num_bytes /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", num_bytes, units[unit_index])
}

pub fn print_memory_usage() {
    let mut system = System::new_all();
    system.refresh_all();

    if let Some(process) = system.process(sysinfo::get_current_pid().unwrap()) {
        let memory_bytes = process.memory();
        let memory_mb = memory_bytes as f64 / 1_048_576.0; // 1 MB = 1,048,576 B
        println!("Memory used by the process: {memory_mb:.2} MB");
    } else {
        println!("Could not get process information.");
    }
}

pub fn create_pool(n_cores: usize) -> ThreadPool {
    ThreadPoolBuilder::new().num_threads(n_cores).build().unwrap()
}

pub fn configured_num_threads(n_local_processes: usize) -> usize {
    let num_cores = num_cpus::get_physical();
    tracing::info!("Node has {num_cores} cores");
    if let Ok(val) = env::var("RAYON_NUM_THREADS") {
        match val.parse::<usize>() {
            Ok(n) if n > 0 => {
                tracing::info!("Using {n} threads per process based on RAYON_NUM_THREADS environment variable");
                return n;
            }
            _ => eprintln!("Warning: RAYON_NUM_THREADS=\"{val}\" invalid, falling back to physical cores"),
        }
    }

    let num = num_cpus::get_physical() / n_local_processes;
    tracing::info!("Using {num} threads based on physical cores per process, considering there are {n_local_processes} processes per node");
    num
}

pub fn join_thread(handle: std::thread::JoinHandle<ProofmanResult<()>>) -> ProofmanResult<()> {
    match handle.join() {
        Ok(inner_result) => inner_result, // propagate closure error
        Err(panic_info) => {
            // Try to get a string from the panic payload
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown thread panic".to_string()
            };
            Err(ProofmanError::ProofmanError(panic_msg))
        }
    }
}

pub fn init_gpu_setup(max_n_bits_ext: u64, gpu: bool) -> ProofmanResult<()> {
    if !set_gpu_mode_c(gpu) {
        return Err(ProofmanError::InvalidConfiguration(
            "GPU mode requested but library was built without CUDA support".into(),
        ));
    }
    if gpu {
        let n_gpus = get_num_gpus_c();
        if n_gpus == 0 {
            return Err(ProofmanError::InvalidConfiguration("No GPUs found".into()));
        }

        init_gpu_setup_c(max_n_bits_ext, GOLDILOCKS_MERKLE_TREE_ARITY);
    }
    Ok(())
}
