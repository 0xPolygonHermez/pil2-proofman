// extern crate env_logger;
use clap::Parser;
use pil2_pilout::{
    pilout::{SymbolType},
    pilout_proxy::PilOutProxy,
};
use pil2_pilout::pilout::hint_field::Value::HintFieldArray;
use pil2_pilout::pilout::hint_field::Value::StringValue;
use pil2_pilout::pilout::hint_field::Value::Operand;
use pil2_pilout::pilout::operand::Operand::Constant;
use proofman_common::initialize_logger;
use proofman_pilfflonk::BN254_R;
use num_bigint::BigUint;
use serde::Serialize;
use tinytemplate::TinyTemplate;
use std::{fs, path::PathBuf};
use colored::Colorize;
use convert_case::{Case, Casing};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct PilHelpersCmd {
    #[clap(long)]
    pub pilout: PathBuf,

    #[clap(long)]
    pub path: PathBuf,

    #[clap(short)]
    pub overide: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

#[derive(Clone, Debug, Serialize)]
struct PackInfo {
    airgroup_id: usize,
    air_id: usize,
    is_packed: bool,
    num_packed_words: u64,
    unpack_info: String,
}

#[derive(Clone, Serialize)]
struct ProofCtx {
    project_name: String,
    num_stages: u32,
    pilout_filename: String,
    pilout_hash: String,
    air_groups: Vec<AirGroupsCtx>,
    constant_airgroups: Vec<(String, usize)>,
    constant_airs: Vec<(String, usize, Vec<usize>, String)>,
    proof_values: Vec<ValuesCtx>,
    publics: Vec<ValuesCtx>,
    has_packed: bool,
    packed_info: Vec<PackInfo>,
    /// The pilout is over BN254's `Fr` ([`is_bn254`]).
    is_bn254: bool,
}

#[derive(Clone, Debug, Serialize)]
struct AirGroupsCtx {
    airgroup_id: usize,
    name: String,
    snake_name: String,
    airs: Vec<AirCtx>,
}

#[derive(Clone, Debug, Serialize)]
struct AirCtx {
    id: usize,
    name: String,
    num_rows: u32,
    has_packed: bool,
    columns: Vec<ColumnCtx>,
    fixed: Vec<ColumnCtx>,
    stages_columns: Vec<StageColumnCtx>,
    custom_columns: Vec<CustomCommitsCtx>,
    air_values: Vec<ValuesCtx>,
    airgroup_values: Vec<ValuesCtx>,
}

#[derive(Clone, Debug, Serialize)]
struct ValuesCtx {
    values: Vec<ColumnCtx>,
    values_u64: Vec<Column64Ctx>,
    values_default: Vec<ColumnCtx>,
}

#[derive(Clone, Debug, Serialize)]
struct CustomCommitsCtx {
    name: String,
    commit_id: usize,
    custom_columns: Vec<ColumnCtx>,
}
#[derive(Clone, Debug, Serialize)]
struct ColumnCtx {
    name: String,
    r#type: String,
    type_packed: String,
}

#[derive(Clone, Debug, Serialize)]
struct Column64Ctx {
    name: String,
    array: bool,
    r#type: String,
    r#type_default: String,
}

#[derive(Default, Clone, Debug, Serialize)]
struct StageColumnCtx {
    stage_id: usize,
    columns: Vec<ColumnCtx>,
}

/// Whether `pilout` is over BN254's `Fr`, the field of `setup-pilfflonk`
/// (pilfflonk/docs/README.md#what-the-setup-refuses), from its `baseField` as the pilfflonk setup
/// reads it. Its helpers are then those of a pilfflonk witness library
/// (pilfflonk/docs/README.md#witness): rows over `Bn254` with no packed rows, which pilfflonk does
/// not accept yet; values of dimension 1, as BN254 has no extension field; and publics that are
/// `Bn254` values.
fn is_bn254(pilout: &pil2_pilout::pilout::PilOut) -> bool {
    BigUint::parse_bytes(BN254_R.as_bytes(), 10).is_some_and(|r| BigUint::from_bytes_be(&pilout.base_field) == r)
}

impl PilHelpersCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        initialize_logger(self.verbose.into(), None);

        tracing::info!("{}", format!("{} Pil-helpers", format!("{: >12}", "Command").bright_green().bold()));
        tracing::info!("");

        // Check if the pilout file exists
        if !self.pilout.exists() {
            return Err(format!("Pilout file '{}' does not exist", self.pilout.display()).into());
        }

        // Check if the path exists
        let pil_helpers_path = self.path.join("pil_helpers");
        if !pil_helpers_path.exists() {
            std::fs::create_dir_all(&pil_helpers_path)?;
        } else if !pil_helpers_path.is_dir() {
            return Err(format!("Path '{}' already exists and is not a folder", pil_helpers_path.display()).into());
        }

        let files = ["mod.rs", "pilout.rs"];

        if !self.overide {
            // Check if the files already exist and launch an error if they do
            for file in files.iter() {
                let dst = pil_helpers_path.join(file);
                if dst.exists() {
                    return Err(format!("{} already exists, skipping", dst.display()).into());
                }
            }
        }

        let pilout_data = fs::read(self.pilout.display().to_string())?;
        let pilout_hash = blake3::hash(&pilout_data).to_hex().to_string();

        // Read the pilout file
        let pilout = PilOutProxy::new(&self.pilout.display().to_string())?;
        let is_bn254 = is_bn254(&pilout);
        // BN254 has no extension field: every value has dimension 1.
        let extension = if is_bn254 { "F" } else { "FieldExtension<F>" };
        // The type of a public in the public inputs a library reads, and its zero.
        let (public_type, public_zero) = if is_bn254 { ("Bn254", "Bn254::default()") } else { ("u64", "0") };

        let mut wcctxs = Vec::new();
        let mut constant_airgroups: Vec<(String, usize)> = Vec::new();
        let mut constant_airs: Vec<(String, usize, Vec<usize>, String)> = Vec::new();
        let mut has_packed = false;

        for (airgroup_id, airgroup) in pilout.air_groups.iter().enumerate() {
            wcctxs.push(AirGroupsCtx {
                airgroup_id,
                name: airgroup.name.as_ref().unwrap().to_case(Case::Pascal),
                snake_name: airgroup.name.as_ref().unwrap().to_case(Case::Snake).to_uppercase(),
                airs: airgroup
                    .airs
                    .iter()
                    .enumerate()
                    .map(|(air_id, air)| {
                        let has_witness_bits = pilout.hints.iter().any(|h| {
                            h.air_group_id == Some(airgroup_id as u32)
                                && h.air_id == Some(air_id as u32)
                                && h.name == "witness_bits"
                        });
                        if has_witness_bits {
                            has_packed = true;
                        }
                        AirCtx {
                            id: air_id,
                            name: air.name.as_ref().unwrap().to_string(),
                            num_rows: air.num_rows.unwrap(),
                            has_packed: has_witness_bits,
                            columns: Vec::new(),
                            fixed: Vec::new(),
                            stages_columns: vec![StageColumnCtx::default(); pilout.num_stages() as usize - 1],
                            custom_columns: Vec::new(),
                            air_values: Vec::new(),
                            airgroup_values: Vec::new(),
                        }
                    })
                    .collect(),
            });

            // Prepare constants
            constant_airgroups.push((airgroup.name.as_ref().unwrap().to_case(Case::Snake).to_uppercase(), airgroup_id));

            for (air_idx, air) in airgroup.airs.iter().enumerate() {
                let air_name = air.name.as_ref().unwrap().to_case(Case::Snake).to_uppercase();
                let contains_key = constant_airs.iter().position(|(name, _, _, _)| name == &air_name);

                let idx = contains_key.unwrap_or_else(|| {
                    constant_airs.push((air_name, airgroup_id, Vec::new(), "".to_owned()));
                    constant_airs.len() - 1
                });

                constant_airs[idx].2.push(air_idx);
            }

            for constant in constant_airs.iter_mut() {
                constant.3 = constant.2.iter().map(|&num| num.to_string()).collect::<Vec<String>>().join(",");
            }
        }

        // The packed rows of `witness_bits` are 64-bit, and pilfflonk does not accept packed traces yet.
        if is_bn254 {
            if let Some(air) = wcctxs.iter().flat_map(|airgroup| &airgroup.airs).find(|air| air.has_packed) {
                return Err(format!(
                    "air {} has `witness_bits` hints, from columns declared with `bits(n)`, which ask for packed trace \
                     rows, and pilfflonk does not accept packed traces yet",
                    air.name
                )
                .into());
            }
        }

        let mut publics: Vec<ValuesCtx> = Vec::new();
        let mut proof_values: Vec<ValuesCtx> = Vec::new();

        pilout
            .symbols
            .iter()
            .filter(|symbol| {
                (symbol.r#type == SymbolType::PublicValue as i32) || (symbol.r#type == SymbolType::ProofValue as i32)
            })
            .for_each(|symbol| {
                let name = symbol.name.split_once('.').map(|x| x.1).unwrap_or(&symbol.name);
                let r#type = if symbol.lengths.is_empty() {
                    "F".to_string() // Case when lengths.len() == 0
                } else {
                    // Start with "F" and apply each length in reverse order
                    symbol.lengths.iter().rev().fold("F".to_string(), |acc, &length| format!("[{acc}; {length}]"))
                };
                let ext_type = if symbol.lengths.is_empty() {
                    extension.to_string() // Case when lengths.len() == 0
                } else {
                    // Start with "F" and apply each length in reverse order
                    symbol.lengths.iter().rev().fold(extension.to_string(), |acc, &length| format!("[{acc}; {length}]"))
                };
                if symbol.r#type == SymbolType::ProofValue as i32 {
                    if proof_values.is_empty() {
                        proof_values.push(ValuesCtx {
                            values: Vec::new(),
                            values_u64: Vec::new(),
                            values_default: Vec::new(),
                        });
                    }
                    if symbol.stage == Some(1) {
                        proof_values[0].values.push(ColumnCtx {
                            name: name.to_owned(),
                            r#type,
                            type_packed: String::new(),
                        });
                    } else {
                        proof_values[0].values.push(ColumnCtx {
                            name: name.to_owned(),
                            r#type: ext_type,
                            type_packed: String::new(),
                        });
                    }
                } else {
                    if publics.is_empty() {
                        publics.push(ValuesCtx {
                            values: Vec::new(),
                            values_u64: Vec::new(),
                            values_default: Vec::new(),
                        });
                    }
                    publics[0].values.push(ColumnCtx { name: name.to_owned(), r#type, type_packed: String::new() });
                    let r#type_64 = if symbol.lengths.is_empty() {
                        public_type.to_string() // Case when lengths.len() == 0
                    } else {
                        // Start with the public's type and apply each length in reverse order
                        symbol
                            .lengths
                            .iter()
                            .rev()
                            .fold(public_type.to_string(), |acc, &length| format!("[{acc}; {length}]"))
                    };
                    let default = public_zero.to_string();
                    let r#type_default = if symbol.lengths.is_empty() {
                        default // Case when lengths.len() == 0
                    } else {
                        // Start with "u64" and apply each length in reverse order
                        symbol.lengths.iter().rev().fold(default, |acc, &length| format!("[{acc}; {length}]"))
                    };
                    publics[0].values_u64.push(Column64Ctx {
                        name: name.to_owned(),
                        r#type: r#type_64,
                        r#type_default: r#type_default.clone(),
                        array: !symbol.lengths.is_empty(),
                    });
                    publics[0].values_default.push(ColumnCtx {
                        name: name.to_owned(),
                        r#type: r#type_default,
                        type_packed: String::new(),
                    });
                }
            });

        // Build columns data for traces
        let mut packed_info: Vec<PackInfo> = Vec::new();
        for (airgroup_id, airgroup) in pilout.air_groups.iter().enumerate() {
            for (air_id, _) in airgroup.airs.iter().enumerate() {
                let air = wcctxs[airgroup_id].airs.get_mut(air_id).unwrap();
                let is_packed = air.has_packed;
                air.custom_columns = pilout.air_groups[airgroup_id].airs[air_id]
                    .custom_commits
                    .iter()
                    .enumerate()
                    .map(|(index, commit)| CustomCommitsCtx {
                        name: commit.name.as_ref().unwrap().to_case(Case::Pascal),
                        commit_id: index,
                        custom_columns: Vec::new(),
                    })
                    .collect();

                // Search symbols where airgroup_id == airgroup_id && air_id == air_id && type == WitnessCol
                let mut vec_bits = vec![];
                pilout
                    .symbols
                    .iter()
                    .filter(|symbol| {
                        symbol.air_group_id.is_some()
                            && symbol.air_group_id.unwrap() == airgroup_id as u32
                            && ((symbol.air_id.is_some() && symbol.air_id.unwrap() == air_id as u32)
                                || symbol.r#type == SymbolType::AirGroupValue as i32)
                            && symbol.stage.is_some()
                            && ((symbol.r#type == SymbolType::WitnessCol as i32)
                                || (symbol.r#type == SymbolType::FixedCol as i32)
                                || (symbol.r#type == SymbolType::AirValue as i32)
                                || (symbol.r#type == SymbolType::AirGroupValue as i32)
                                || (symbol.r#type == SymbolType::CustomCol as i32 && symbol.stage.unwrap() == 0))
                    })
                    .for_each(|symbol| {
                        let air = wcctxs[airgroup_id].airs.get_mut(air_id).unwrap();
                        let name = symbol.name.split_once('.').map(|x| x.1).unwrap_or(&symbol.name);
                        let r#type = if symbol.lengths.is_empty() {
                            "F".to_string() // Case when lengths.len() == 0
                        } else {
                            // Start with "F" and apply each length in reverse order
                            symbol
                                .lengths
                                .iter()
                                .rev()
                                .fold("F".to_string(), |acc, &length| format!("[{acc}; {length}]"))
                        };
                        let type_packed = if air.has_packed
                            && symbol.r#type == SymbolType::WitnessCol as i32
                            && symbol.stage.unwrap() == 1
                        {
                            let hint = pilout.hints.iter().find(|h| {
                                h.air_group_id == Some(airgroup_id as u32)
                                    && h.air_id == Some(air_id as u32)
                                    && h.name == "witness_bits"
                                    && h.hint_fields.iter().any(|field| {
                                        if let Some(HintFieldArray(ref array)) = &field.value {
                                            array.hint_fields.iter().any(|inner_field| {
                                                if let (Some(inner_name), Some(StringValue(ref string_val))) =
                                                    (&inner_field.name, &inner_field.value)
                                                {
                                                    inner_name == "name" && string_val == name
                                                } else {
                                                    false
                                                }
                                            })
                                        } else {
                                            false
                                        }
                                    })
                            });

                            // A column may live in a packed air without itself being packed
                            // (e.g. std range-check multiplicity columns like `mul_range`).
                            // Such columns have no `witness_bits` hint; emit them as unpacked.
                            if let Some(hint) = hint {
                                let bits = hint
                                    .hint_fields
                                    .iter()
                                    .find_map(|field| {
                                        if let Some(HintFieldArray(ref array)) = &field.value {
                                            for inner_field in array.hint_fields.iter() {
                                                if let (Some(inner_name), Some(Operand(operand))) =
                                                    (&inner_field.name, &inner_field.value)
                                                {
                                                    if inner_name == "bits" {
                                                        if let Some(Constant(constant)) = &operand.operand {
                                                            if !constant.value.is_empty() {
                                                                return Some(constant.value[0]);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        None
                                    })
                                    .expect("bits not found");

                                let type_bits = match bits {
                                    1 => "bit".to_string(),
                                    8 => "u8".to_string(),
                                    16 => "u16".to_string(),
                                    32 => "u32".to_string(),
                                    64 => "u64".to_string(),
                                    _ => format!("ubit({bits})"), // dynamically include bits
                                };

                                let total_lengths = symbol.lengths.iter().product::<u32>();
                                vec_bits.extend(vec![bits as u64; total_lengths as usize]);
                                if symbol.lengths.is_empty() {
                                    type_bits.to_string()
                                } else {
                                    symbol
                                        .lengths
                                        .iter()
                                        .rev()
                                        .fold(type_bits.to_string(), |acc, &length| format!("[{acc}; {length}]"))
                                }
                            } else {
                                // Column in a packed air with no `witness_bits` hint
                                // (e.g. std range-check columns like `mul_range`):
                                // pack it as a full 64-bit word.
                                let bits = 64u64;
                                let total_lengths = symbol.lengths.iter().product::<u32>();
                                vec_bits.extend(vec![bits; total_lengths as usize]);
                                if symbol.lengths.is_empty() {
                                    "u64".to_string()
                                } else {
                                    symbol
                                        .lengths
                                        .iter()
                                        .rev()
                                        .fold("u64".to_string(), |acc, &length| format!("[{acc}; {length}]"))
                                }
                            }
                        } else {
                            String::new()
                        };
                        let ext_type = if symbol.lengths.is_empty() {
                            extension.to_string() // Case when lengths.len() == 0
                        } else {
                            // Start with "F" and apply each length in reverse order
                            symbol
                                .lengths
                                .iter()
                                .rev()
                                .fold(extension.to_string(), |acc, &length| format!("[{acc}; {length}]"))
                        };
                        if symbol.r#type == SymbolType::WitnessCol as i32 {
                            if symbol.stage.unwrap() == 1 {
                                air.columns.push(ColumnCtx { name: name.to_owned(), r#type, type_packed });
                            } else {
                                air.stages_columns[symbol.stage.unwrap() as usize - 2].stage_id =
                                    symbol.stage.unwrap() as usize;
                                air.stages_columns[symbol.stage.unwrap() as usize - 2].columns.push(ColumnCtx {
                                    name: name.to_owned(),
                                    r#type: ext_type,
                                    type_packed: String::new(),
                                });
                            }
                        } else if symbol.r#type == SymbolType::FixedCol as i32 {
                            air.fixed.push(ColumnCtx { name: name.to_owned(), r#type, type_packed: String::new() });
                        } else if symbol.r#type == SymbolType::AirValue as i32 {
                            if air.air_values.is_empty() {
                                air.air_values.push(ValuesCtx {
                                    values: Vec::new(),
                                    values_u64: Vec::new(),
                                    values_default: Vec::new(),
                                });
                            }
                            if symbol.stage == Some(1) {
                                air.air_values[0].values.push(ColumnCtx {
                                    name: name.to_owned(),
                                    r#type,
                                    type_packed: String::new(),
                                });
                            } else {
                                air.air_values[0].values.push(ColumnCtx {
                                    name: name.to_owned(),
                                    r#type: ext_type,
                                    type_packed: String::new(),
                                });
                            }
                        } else if symbol.r#type == SymbolType::AirGroupValue as i32 {
                            if air.airgroup_values.is_empty() {
                                air.airgroup_values.push(ValuesCtx {
                                    values: Vec::new(),
                                    values_u64: Vec::new(),
                                    values_default: Vec::new(),
                                });
                            }
                            air.airgroup_values[0].values.push(ColumnCtx {
                                name: name.to_owned(),
                                r#type: ext_type,
                                type_packed: String::new(),
                            });
                        } else {
                            air.custom_columns[symbol.commit_id.unwrap() as usize].custom_columns.push(ColumnCtx {
                                name: name.to_owned(),
                                r#type,
                                type_packed: String::new(),
                            });
                        }
                    });
                if is_packed {
                    let num_packed_words = vec_bits.iter().sum::<u64>().div_ceil(64);
                    let unpack_info = vec_bits.iter().map(|b| b.to_string()).collect::<Vec<String>>().join(", ");
                    packed_info.push(PackInfo { airgroup_id, air_id, is_packed: true, num_packed_words, unpack_info });
                }
            }
        }

        let context = ProofCtx {
            project_name: pilout.name.as_ref().unwrap().to_case(Case::Pascal),
            pilout_hash,
            num_stages: pilout.num_stages(),
            pilout_filename: self.pilout.file_name().unwrap().to_str().unwrap().to_string(),
            air_groups: wcctxs,
            constant_airs,
            constant_airgroups,
            publics,
            proof_values,
            has_packed,
            packed_info,
            is_bn254,
        };

        const MOD_RS: &str = include_str!("../../assets/templates/pil_helpers_mod.rs.tt");

        let mut tt = TinyTemplate::new();
        tt.add_template("mod.rs", MOD_RS)?;
        tt.add_template("traces.rs", include_str!("../../assets/templates/pil_helpers_trace.rs.tt"))?;

        // Write the files
        // --------------------------------------------
        // Write mod.rs
        fs::write(pil_helpers_path.join("mod.rs"), MOD_RS)?;

        // Write traces.rs
        fs::write(
            pil_helpers_path.join("traces.rs"),
            tt.render("traces.rs", &context)?.replace("&lt;", "<").replace("&gt;", ">"),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pil2_pilout::pilout::{self as pb, hint_field};
    use prost::Message;

    /// Goldilocks, `2^64 − 2^32 + 1`.
    const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;

    /// A fresh directory of its own for a test, removed when it is dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(test: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("proofman_cli_pil_helpers_{test}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn symbol(name: &str, kind: SymbolType, stage: u32, lengths: &[u32]) -> pb::Symbol {
        let of_air = !matches!(kind, SymbolType::PublicValue | SymbolType::ProofValue);
        pb::Symbol {
            name: name.to_string(),
            air_group_id: of_air.then_some(0),
            air_id: (of_air && kind != SymbolType::AirGroupValue).then_some(0),
            r#type: kind as i32,
            stage: Some(stage),
            lengths: lengths.to_vec(),
            ..Default::default()
        }
    }

    /// A pilout of one AIR, `Main`, of stage 1 only (no challenges, as a program without the std's
    /// buses has), with every kind of value pil-helpers writes: witness and fixed columns, publics,
    /// an air value, an airgroup value and a proof value of stage 2, over the field `base_field`.
    fn pilout(base_field: Vec<u8>) -> pb::PilOut {
        let air = pb::Air { name: Some("Main".into()), num_rows: Some(8), stage_widths: vec![3], ..Default::default() };
        pb::PilOut {
            name: Some("Program".into()),
            base_field,
            air_groups: vec![pb::AirGroup { name: Some("Group".into()), airs: vec![air], ..Default::default() }],
            symbols: vec![
                symbol("Main.a", SymbolType::WitnessCol, 1, &[]),
                symbol("Main.b", SymbolType::WitnessCol, 1, &[2]),
                symbol("Main.L1", SymbolType::FixedCol, 0, &[]),
                symbol("p", SymbolType::PublicValue, 1, &[]),
                symbol("q", SymbolType::PublicValue, 1, &[2]),
                symbol("Main.av", SymbolType::AirValue, 2, &[]),
                symbol("Group.agv", SymbolType::AirGroupValue, 2, &[]),
                symbol("pv", SymbolType::ProofValue, 2, &[]),
            ],
            ..Default::default()
        }
    }

    fn bn254() -> Vec<u8> {
        BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap().to_bytes_be()
    }

    /// The `witness_bits` hint `col witness bits(8) a` gives.
    fn witness_bits() -> pb::Hint {
        let field = |name: &str, value| pb::HintField { name: Some(name.to_string()), value: Some(value) };
        let bits =
            pb::Operand { operand: Some(pb::operand::Operand::Constant(pb::operand::Constant { value: vec![8] })) };
        let fields = vec![field("name", hint_field::Value::StringValue("a".into())), field("bits", Operand(bits))];
        pb::Hint {
            name: "witness_bits".into(),
            hint_fields: vec![pb::HintField {
                name: None,
                value: Some(HintFieldArray(pb::HintFieldArray { hint_fields: fields })),
            }],
            air_group_id: Some(0),
            air_id: Some(0),
        }
    }

    /// Runs pil-helpers on `pilout` in `dir`, and returns its `traces.rs`. One test at a time: the
    /// command sets up the global logger, which two threads cannot both do.
    fn pil_helpers(dir: &TempDir, pilout: &pb::PilOut) -> Result<String, String> {
        static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _serial = SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let path = dir.0.join("program.pilout");
        fs::write(&path, pilout.encode_to_vec()).unwrap();
        let out = dir.0.join("src");
        let cmd = PilHelpersCmd { pilout: path, path: out.clone(), overide: true, verbose: 0 };
        cmd.run().map_err(|e| e.to_string())?;
        Ok(fs::read_to_string(out.join("pil_helpers").join("traces.rs")).unwrap())
    }

    #[test]
    fn a_bn254_pilout_gives_rows_over_bn254() {
        let dir = TempDir::new("bn254");
        let traces = pil_helpers(&dir, &pilout(bn254())).unwrap();

        // No extension field: the values of stage 2 have dimension 1.
        assert!(!traces.contains("FieldExtension"), "{traces}");
        for value in [" av: F,", " agv: F,", " pv: F,"] {
            assert!(traces.contains(value), "{value}: {traces}");
        }
        // The publics a library reads are Bn254 values, zero by default.
        assert!(traces.contains("use proofman_fields::Bn254;\n"), "{traces}");
        assert!(traces.contains("    pub p: Bn254,\n") && traces.contains("    pub q: [Bn254; 2],\n"), "{traces}");
        assert!(traces.contains("fn default_array_q() -> [Bn254; 2] {\n    [Bn254::default(); 2]\n}"), "{traces}");
        assert!(traces.contains("values!(ProgramPublicValues<F> {\n p: F, q: [F; 2],\n});"), "{traces}");
        // Plain rows over the pilout's columns, and no packed trace.
        assert!(traces.contains("trace_row!(MainTraceRow<F> {\n a:F, b:[F; 2],\n});"), "{traces}");
        assert!(traces.contains("pub type MainTrace<F> = GenericTrace<MainTraceRow<F>, 8, 0, 0>;"), "{traces}");
        assert!(!traces.contains("PACKED_INFO") && !traces.contains("PackedInfoConst"), "{traces}");
        assert!(traces.contains("(0, 0, \"Main\"),"), "{traces}");
    }

    /// The same program over Goldilocks keeps its extension field, its `u64` publics and its
    /// `PACKED_INFO`: the STARK's helpers do not change.
    #[test]
    fn a_goldilocks_pilout_keeps_the_starks_helpers() {
        let dir = TempDir::new("goldilocks");
        let traces = pil_helpers(&dir, &pilout(BigUint::from(GOLDILOCKS).to_bytes_be())).unwrap();
        assert!(traces.contains("#[allow(dead_code)]\ntype FieldExtension<F> = [F; 3];\n"), "{traces}");
        for value in [" av: FieldExtension<F>,", " agv: FieldExtension<F>,", " pv: FieldExtension<F>,"] {
            assert!(traces.contains(value), "{value}: {traces}");
        }
        assert!(traces.contains("    pub p: u64,\n") && traces.contains("    pub q: [u64; 2],\n"), "{traces}");
        assert!(!traces.contains("Bn254"), "{traces}");
        assert!(traces.contains("use proofman_common::PackedInfoConst;\n"), "{traces}");
        assert!(traces.contains("pub const PACKED_INFO: &[(usize, usize, PackedInfoConst)] = &[\n];"), "{traces}");
    }

    /// pilfflonk does not accept packed traces yet: a BN254 pilout with a `witness_bits` hint is
    /// refused, and nothing is written.
    #[test]
    fn a_bn254_pilout_with_packed_columns_is_refused() {
        let dir = TempDir::new("bn254_packed");
        let mut pilout = pilout(bn254());
        pilout.hints.push(witness_bits());
        let err = pil_helpers(&dir, &pilout).unwrap_err();
        assert!(err.contains("air Main") && err.contains("pilfflonk does not accept packed traces yet"), "{err}");
        assert!(!dir.0.join("src").join("pil_helpers").join("traces.rs").exists());

        // Over Goldilocks the same hint packs the column.
        pilout.base_field = BigUint::from(GOLDILOCKS).to_bytes_be();
        let traces = pil_helpers(&dir, &pilout).unwrap();
        assert!(traces.contains("trace_row!(MainTraceRow<F> {\n a:u8, b:[u64; 2],\n});"), "{traces}");
        assert!(traces.contains("pub type MainTrace<R> = GenericTrace<R, 8, 0, 0>;"), "{traces}");
    }
}
