use crate::{ProofmanError, ProofmanResult, DEFAULT_N_PRINT_CONSTRAINTS};
use crate::{
    AirGroupMap, AirIdMap, DebugInfo, GlobalInfo, InstanceMap, ModeName, ProofCtx, StdMode, VerboseMode,
    DEFAULT_PRINT_VALS,
};
use proofman_starks_lib_c::{set_log_level_c, init_gpu_setup_c, get_num_gpus_c};
use tracing::dispatcher;
use tracing_subscriber::filter::LevelFilter;
use std::path::PathBuf;
use std::collections::HashMap;
use fields::PrimeField64;
use serde::Deserialize;
use std::fs;
use sysinfo::System;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use tracing_subscriber::prelude::*;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::format::FormatFields;
use tracing_subscriber::fmt::time::SystemTime;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::fmt::FormatEvent;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt;
use std::sync::OnceLock;
use yansi::Color;
use yansi::Paint;
use std::io::IsTerminal;
use colored::Colorize;
use crate::RankInfo;

static GLOBAL_RANK: OnceLock<i32> = OnceLock::new();
static VERBOSE_MODE: OnceLock<VerboseMode> = OnceLock::new();
static IS_TERMINAL: OnceLock<bool> = OnceLock::new();

use crate::InstancesInfo;

pub struct RankFormatter;

impl<S, N> FormatEvent<S, N> for RankFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &fmt::FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let timer = SystemTime;

        let mut time_str = String::new();
        {
            let mut fake_writer = Writer::new(&mut time_str);
            timer.format_time(&mut fake_writer)?;
        }

        let is_terminal = IS_TERMINAL.get().copied().unwrap_or(false);

        if is_terminal {
            write!(writer, "{} ", time_str.dimmed())?;
        } else {
            write!(writer, "{time_str} ")?;
        }

        if let Some(rank) = GLOBAL_RANK.get().copied() {
            let rank_str = match is_terminal {
                true => format!("[rank={rank}]").dimmed(),
                false => format!("[rank={rank}]").into(),
            };
            write!(writer, "{rank_str} ")?;
        }

        let target = event.metadata().target();
        let show_target =
            VERBOSE_MODE.get().map(|vm| matches!(vm, VerboseMode::Debug | VerboseMode::Trace)).unwrap_or(false);

        if is_terminal {
            if show_target {
                write!(writer, "{} ", target.dimmed())?;
            }

            let level_str = match *event.metadata().level() {
                tracing::Level::TRACE => "TRACE".paint(Color::Cyan),
                tracing::Level::DEBUG => "DEBUG".paint(Color::Blue),
                tracing::Level::INFO => "INFO".paint(Color::Green),
                tracing::Level::WARN => "WARN".paint(Color::Yellow),
                tracing::Level::ERROR => "ERROR".paint(Color::Red),
            };
            write!(writer, "{level_str}: ")?;
        } else {
            let level_str = match *event.metadata().level() {
                tracing::Level::TRACE => "TRACE",
                tracing::Level::DEBUG => "DEBUG",
                tracing::Level::INFO => "INFO",
                tracing::Level::WARN => "WARN",
                tracing::Level::ERROR => "ERROR",
            };
            if show_target {
                write!(writer, "{target} ")?;
            }
            write!(writer, "{level_str}: ")?;
        }

        let mut visitor = MessageVisitor::new();
        event.record(&mut visitor);
        write!(writer, "{}", visitor.message)?;
        writeln!(writer)?;

        Ok(())
    }
}

// Add this visitor struct
struct MessageVisitor {
    message: String,
}

impl MessageVisitor {
    fn new() -> Self {
        Self { message: String::new() }
    }
}

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
            if self.message.starts_with('"') && self.message.ends_with('"') {
                self.message = self.message[1..self.message.len() - 1].to_string();
            }
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }
}

pub fn set_global_rank(rank: i32) {
    let _ = GLOBAL_RANK.set(rank);
}

pub fn initialize_logger(verbose_mode: VerboseMode, rank: Option<&RankInfo>) {
    if GLOBAL_RANK.get().is_none() {
        if let Some(r) = rank {
            if r.n_processes > 1 {
                set_global_rank(r.world_rank);
            }
        }
    }

    let _ = VERBOSE_MODE.set(verbose_mode);

    let is_terminal = std::io::stdout().is_terminal() && std::env::var("NO_COLOR").is_err();

    let _ = IS_TERMINAL.set(is_terminal);

    // Disable ANSI/colors globally when not in a terminal
    if !is_terminal {
        yansi::disable();
    }

    if dispatcher::has_been_set() {
        return;
    }

    let stdout_layer = tracing_subscriber::fmt::layer()
        .event_format(RankFormatter)
        .with_writer(std::io::stdout)
        .with_ansi(is_terminal)
        .with_filter(LevelFilter::from(verbose_mode));

    tracing_subscriber::registry().with(stdout_layer).init();

    set_log_level_c(verbose_mode.into());
}

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

pub fn skip_prover_instance<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    global_idx: usize,
) -> ProofmanResult<(bool, Option<InstancesInfo>)> {
    if !pctx.debug_info.read().unwrap().skip_prover_instances {
        return Ok((false, None));
    }

    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(global_idx)?;
    let air_instance_id = pctx.dctx_find_air_instance_id(global_idx)?;

    if let Some(airgroup_id_map) = pctx.debug_info.read().unwrap().debug_instances.get(&airgroup_id) {
        if airgroup_id_map.is_empty() {
            return Ok((false, None));
        } else if let Some(air_id_map) = airgroup_id_map.get(&air_id) {
            if air_id_map.1.is_empty() {
                return Ok((false, None));
            } else if let Some(instance_id_map) = air_id_map.1.get(&air_instance_id) {
                return Ok((false, Some(instance_id_map.clone())));
            }
        }
    }

    Ok((true, None))
}

pub fn store_rows_info_air<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    instance_id: usize,
) -> bool {
    if pctx.debug_info.read().unwrap().store_row_info {
        return true;
    }

    if let Some(airgroup_id_map) = pctx.debug_info.read().unwrap().debug_instances.get(&airgroup_id) {
        if let Some(air_id_map) = airgroup_id_map.get(&air_id) {
            if air_id_map.0 {
                return true;
            }

            if let Some(instance_id_map) = air_id_map.1.get(&instance_id) {
                if instance_id_map.store_row_info {
                    return true;
                }
            }
        }
    }

    false
}

fn default_fast_mode() -> bool {
    true
}

#[derive(Debug, Default, Deserialize)]
struct StdDebugMode {
    #[serde(default)]
    opids: Option<Vec<u64>>,
    #[serde(default)]
    n_vals: Option<usize>,
    #[serde(default)]
    print_to_file: bool,
    #[serde(default = "default_fast_mode")]
    fast_mode: bool,
    #[serde(default)]
    debug_values: Option<Vec<Vec<String>>>,
}

#[derive(Debug, Deserialize)]
struct DebugJson {
    #[serde(default)]
    instances: Option<Vec<AirGroupJson>>,
    #[serde(default)]
    global_constraints: Option<Vec<usize>>,
    #[serde(default)]
    std_mode: Option<StdDebugMode>,
    #[serde(default)]
    n_print_constraints: Option<usize>,
    #[serde(default)]
    skip_prover_instances: Option<bool>,
    #[serde(default)]
    store_row_info: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct AirGroupJson {
    #[serde(default)]
    airgroup_id: Option<usize>,
    #[serde(default)]
    airgroup: Option<String>,
    #[serde(default)]
    air_ids: Option<Vec<AirIdJson>>,
}

#[derive(Debug, Deserialize)]
struct AirIdJson {
    #[serde(default)]
    air_id: Option<usize>,
    #[serde(default)]
    air: Option<String>,
    #[serde(default)]
    instance_ids: Option<Vec<InstanceJson>>,
    #[serde(default)]
    store_row_info: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct InstanceJson {
    #[serde(default)]
    instance_id: Option<usize>,
    #[serde(default)]
    constraints: Option<Vec<usize>>,
    #[serde(default)]
    hint_ids: Option<Vec<usize>>,
    #[serde(default)]
    rows: Option<Vec<usize>>,
    #[serde(default)]
    store_row_info: Option<bool>,
}

pub fn json_to_debug_instances_map(proving_key_path: PathBuf, json_path: String) -> ProofmanResult<DebugInfo> {
    // Check proving_key_path exists
    if !proving_key_path.exists() {
        return Err(ProofmanError::InvalidConfiguration(format!(
            "Proving key folder not found at path: {proving_key_path:?}"
        )));
    }

    let global_info: GlobalInfo = GlobalInfo::new(&proving_key_path)?;

    // Read the file contents
    let debug_json = fs::read_to_string(&json_path)?;

    // Deserialize the JSON into the `DebugJson` struct
    let json: DebugJson = serde_json::from_str(&debug_json)?;

    // Initialize the airgroup map
    let mut airgroup_map: AirGroupMap = HashMap::new();

    // Populate the airgroup map using the deserialized data
    if let Some(instances) = json.instances {
        for airgroup in instances {
            let mut air_id_map: AirIdMap = HashMap::new();

            if airgroup.airgroup.is_none() && airgroup.airgroup_id.is_none() {
                return Err(ProofmanError::InvalidSetup(
                    "Airgroup or airgroup_id must be defined in the JSON file".to_string(),
                ));
            }
            if airgroup.airgroup.is_some() && airgroup.airgroup_id.is_some() {
                return Err(ProofmanError::InvalidSetup(
                    "Only airgroup or airgroup_id can be defined in the JSON file, not both".to_string(),
                ));
            }

            let airgroup_id = if let Some(id) = airgroup.airgroup_id {
                id
            } else {
                let airgroup_name = airgroup.airgroup.unwrap().to_string();
                let airgroup_id = global_info.air_groups.iter().position(|x| x == &airgroup_name);
                if airgroup_id.is_none() {
                    return Err(ProofmanError::InvalidSetup(format!(
                        "Airgroup name {airgroup_name} not found in global_info.airgroups"
                    )));
                }
                airgroup_id.unwrap()
            };

            if let Some(air_ids) = airgroup.air_ids {
                for air in air_ids {
                    if air.air.is_none() && air.air_id.is_none() {
                        return Err(ProofmanError::InvalidSetup(
                            "Air or air_id must be defined in the JSON file".to_string(),
                        ));
                    }
                    if air.air.is_some() && air.air_id.is_some() {
                        return Err(ProofmanError::InvalidSetup(
                            "Only air or air_id can be defined in the JSON file, not both".to_string(),
                        ));
                    }

                    let air_id = if let Some(id) = air.air_id {
                        id
                    } else {
                        let air_name = air.air.unwrap().to_string();
                        let air_id = global_info.airs[airgroup_id].iter().position(|x| x.name == air_name);
                        if air_id.is_none() {
                            return Err(ProofmanError::InvalidSetup(format!(
                                "Airgroup name {air_name} not found in global_info.airgroups"
                            )));
                        }
                        air_id.unwrap()
                    };

                    let mut instance_map: InstanceMap = HashMap::new();

                    if let Some(instances) = air.instance_ids {
                        for instance in instances {
                            let instance_constraints = instance.constraints.unwrap_or_default();
                            let hint_ids = instance.hint_ids.unwrap_or_default();
                            let rows = instance.rows.unwrap_or_default();
                            let store_row_info = instance.store_row_info.unwrap_or(false);
                            instance_map.insert(
                                instance.instance_id.unwrap_or_default(),
                                InstancesInfo { constraints: instance_constraints, hint_ids, rows, store_row_info },
                            );
                        }
                    }

                    air_id_map.insert(air_id, (air.store_row_info.unwrap_or(false), instance_map));
                }
            }

            airgroup_map.insert(airgroup_id, air_id_map);
        }
    }

    // Default global_constraints to an empty Vec if None
    let global_constraints = json.global_constraints.unwrap_or_default();

    let std_mode = if json.std_mode.is_none() {
        StdMode::new(ModeName::Standard, Vec::new(), 0, false, false, Vec::new())
    } else {
        let mode = json.std_mode.unwrap_or_default();
        let fast_mode =
            if mode.opids.is_some() && !mode.opids.as_ref().unwrap().is_empty() { false } else { mode.fast_mode };

        let debug_values = mode.debug_values.unwrap_or_default();

        StdMode::new(
            ModeName::Debug,
            mode.opids.unwrap_or_default(),
            mode.n_vals.unwrap_or(DEFAULT_PRINT_VALS),
            mode.print_to_file,
            fast_mode,
            debug_values,
        )
    };

    let n_print_constraints = json.n_print_constraints.unwrap_or(DEFAULT_N_PRINT_CONSTRAINTS);
    let skip_prover_instances = json.skip_prover_instances.unwrap_or(false);
    let store_row_info = json.store_row_info.unwrap_or(false);
    Ok(DebugInfo {
        debug_instances: airgroup_map.clone(),
        debug_global_instances: global_constraints,
        std_mode,
        n_print_constraints,
        skip_prover_instances,
        store_row_info,
    })
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

pub fn init_gpu_setup(max_n_bits_ext: u64) -> ProofmanResult<()> {
    if cfg!(feature = "gpu") {
        let n_gpus = get_num_gpus_c();
        if n_gpus == 0 {
            return Err(ProofmanError::InvalidConfiguration("No GPUs found".into()));
        }

        init_gpu_setup_c(max_n_bits_ext);
    }
    Ok(())
}
