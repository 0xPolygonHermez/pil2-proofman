use crate::DEFAULT_N_PRINT_CONSTRAINTS;
use crate::{
    AirGroupMap, AirIdMap, DebugInfo, GlobalInfo, InstanceMap, ModeName, ProofCtx, StdMode, VerboseMode,
    DEFAULT_PRINT_VALS,
};
use proofman_starks_lib_c::set_log_level_c;
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
use colored::Colorize;
use std::io::IsTerminal;

static GLOBAL_RANK: OnceLock<i32> = OnceLock::new();

struct RankFormatter;

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

        if std::io::stdout().is_terminal() {
            write!(writer, "{} ", time_str.dimmed())?;
        } else {
            write!(writer, "{} ", time_str)?;
        }

        if let Some(rank) = GLOBAL_RANK.get().copied() {
            let rank_str = match std::io::stdout().is_terminal() {
                true => format!("[rank={rank}]").dimmed(),
                false => format!("[rank={rank}]").into(),
            };
            write!(writer, "{} ", rank_str)?;
        }

        if std::io::stdout().is_terminal() {
            let level_str = match *event.metadata().level() {
                tracing::Level::TRACE => "TRACE".paint(Color::Cyan),
                tracing::Level::DEBUG => "DEBUG".paint(Color::Blue),
                tracing::Level::INFO => "INFO".paint(Color::Green),
                tracing::Level::WARN => "WARN".paint(Color::Yellow),
                tracing::Level::ERROR => "ERROR".paint(Color::Red),
            };
            write!(writer, "{}: ", level_str)?;
        } else {
            write!(writer, "{}: ", event.metadata().level())?;
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
            self.message = format!("{:?}", value);
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

pub fn initialize_logger(verbose_mode: VerboseMode, rank: Option<i32>) {
    if dispatcher::has_been_set() {
        return;
    }

    if let Some(r) = rank {
        set_global_rank(r);
    }

    let stdout_layer = tracing_subscriber::fmt::layer()
        .event_format(RankFormatter)
        .with_writer(std::io::stdout)
        .with_ansi(false)
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
) -> Result<(bool, Vec<usize>), Box<dyn std::error::Error + Send + Sync>> {
    if pctx.debug_info.read().unwrap().debug_instances.is_empty() {
        return Ok((false, Vec::new()));
    }

    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(global_idx)?;
    let air_instance_id = pctx.dctx_find_air_instance_id(global_idx)?;

    if let Some(airgroup_id_map) = pctx.debug_info.read().unwrap().debug_instances.get(&airgroup_id) {
        if airgroup_id_map.is_empty() {
            return Ok((false, Vec::new()));
        } else if let Some(air_id_map) = airgroup_id_map.get(&air_id) {
            if air_id_map.is_empty() {
                return Ok((false, Vec::new()));
            } else if let Some(instance_id_map) = air_id_map.get(&air_instance_id) {
                return Ok((false, instance_id_map.clone()));
            }
        }
    }

    Ok((true, Vec::new()))
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
}

#[derive(Debug, Deserialize)]
struct DebugJson {
    #[serde(default)]
    constraints: Option<Vec<AirGroupJson>>,
    #[serde(default)]
    global_constraints: Option<Vec<usize>>,
    #[serde(default)]
    std_mode: Option<StdDebugMode>,
    #[serde(default)]
    n_print_constraints: Option<usize>,
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
}

#[derive(Debug, Deserialize)]
struct InstanceJson {
    #[serde(default)]
    instance_id: Option<usize>,
    #[serde(default)]
    constraints: Option<Vec<usize>>,
}

pub fn json_to_debug_instances_map(
    proving_key_path: PathBuf,
    json_path: String,
) -> Result<DebugInfo, Box<dyn std::error::Error + Send + Sync>> {
    // Check proving_key_path exists
    if !proving_key_path.exists() {
        return Err(format!("Proving key folder not found at path: {proving_key_path:?}").into());
    }

    let global_info: GlobalInfo = GlobalInfo::new(&proving_key_path)?;

    // Read the file contents
    let debug_json = fs::read_to_string(&json_path).map_err(|_| format!("Failed to read file {}", json_path))?;

    // Deserialize the JSON into the `DebugJson` struct
    let json: DebugJson =
        serde_json::from_str(&debug_json).map_err(|err| format!("Failed to parse JSON file: {json_path}: {err}"))?;

    // Initialize the airgroup map
    let mut airgroup_map: AirGroupMap = HashMap::new();

    // Populate the airgroup map using the deserialized data
    if let Some(constraints) = json.constraints {
        for airgroup in constraints {
            let mut air_id_map: AirIdMap = HashMap::new();

            if airgroup.airgroup.is_none() && airgroup.airgroup_id.is_none() {
                return Err(("Airgroup or airgroup_id must be defined in the JSON file".to_string()).into());
            }
            if airgroup.airgroup.is_some() && airgroup.airgroup_id.is_some() {
                return Err(
                    ("Only airgroup or airgroup_id can be defined in the JSON file, not both".to_string()).into()
                );
            }

            let airgroup_id = if airgroup.airgroup_id.is_some() {
                airgroup.airgroup_id.unwrap()
            } else {
                let airgroup_name = airgroup.airgroup.unwrap().to_string();
                let airgroup_id = global_info.air_groups.iter().position(|x| x == &airgroup_name);
                if airgroup_id.is_none() {
                    return Err(format!("Airgroup name {airgroup_name} not found in global_info.airgroups").into());
                }
                airgroup_id.unwrap()
            };

            if let Some(air_ids) = airgroup.air_ids {
                for air in air_ids {
                    if air.air.is_none() && air.air_id.is_none() {
                        return Err(("Air or air_id must be defined in the JSON file".to_string()).into());
                    }
                    if air.air.is_some() && air.air_id.is_some() {
                        return Err(("Only air or air_id can be defined in the JSON file, not both".to_string()).into());
                    }

                    let air_id = if air.air_id.is_some() {
                        air.air_id.unwrap()
                    } else {
                        let air_name = air.air.unwrap().to_string();
                        let air_id = global_info.airs[airgroup_id].iter().position(|x| x.name == air_name);
                        if air_id.is_none() {
                            return Err(format!("Airgroup name {air_name} not found in global_info.airgroups").into());
                        }
                        air_id.unwrap()
                    };

                    let mut instance_map: InstanceMap = HashMap::new();

                    if let Some(instances) = air.instance_ids {
                        for instance in instances {
                            let instance_constraints = instance.constraints.unwrap_or_default();
                            instance_map.insert(instance.instance_id.unwrap_or_default(), instance_constraints);
                        }
                    }

                    air_id_map.insert(air_id, instance_map);
                }
            }

            airgroup_map.insert(airgroup_id, air_id_map);
        }
    }

    // Default global_constraints to an empty Vec if None
    let global_constraints = json.global_constraints.unwrap_or_default();

    let std_mode = if !airgroup_map.is_empty() {
        StdMode::new(ModeName::Standard, Vec::new(), 0, false, false)
    } else {
        let mode = json.std_mode.unwrap_or_default();
        let fast_mode =
            if mode.opids.is_some() && !mode.opids.as_ref().unwrap().is_empty() { false } else { mode.fast_mode };

        StdMode::new(
            ModeName::Debug,
            mode.opids.unwrap_or_default(),
            mode.n_vals.unwrap_or(DEFAULT_PRINT_VALS),
            mode.print_to_file,
            fast_mode,
        )
    };

    let n_print_constraints = json.n_print_constraints.unwrap_or(DEFAULT_N_PRINT_CONSTRAINTS);
    Ok(DebugInfo {
        debug_instances: airgroup_map.clone(),
        debug_global_instances: global_constraints,
        std_mode,
        n_print_constraints,
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

pub fn join_thread(
    handle: std::thread::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync + Send + Sync>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync + Send + Sync>> {
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
            Err(panic_msg.into())
        }
    }
}
