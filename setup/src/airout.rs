//! setup/src/airout.rs
//! ───────────────────
//! Thin wrapper around `PilOutProxy`, plus a light-weight cache that provides
//! per-air `symbols` and `hints`, matching the behaviour of the JavaScript
//! implementation.  No mutation of the generated protobuf structs occurs.

use std::{error::Error, path::Path};

use pilout::{
    pilout::{Hint, PilOut, Symbol},
    pilout_proxy::PilOutProxy,
};

/* ────────────────────────── JS constant mirrors ────────────────────────── */

pub mod consts {
    pub const IM_COL: i32 = 0;
    pub const FIXED_COL: i32 = 1;
    pub const PERIODIC_COL: i32 = 2;
    pub const WITNESS_COL: i32 = 3;
    pub const PROOF_VALUE: i32 = 4;
    pub const AIRGROUP_VALUE: i32 = 5;
    pub const PUBLIC_VALUE: i32 = 6;
    pub const PUBLIC_TABLE: i32 = 7;
    pub const CHALLENGE: i32 = 8;
}

/* ────────────────────────────── caches ─────────────────────────────────── */

#[derive(Debug)]
struct Caches {
    symbols_by_air: Vec<Vec<Vec<Symbol>>>,
    hints_by_air: Vec<Vec<Vec<Hint>>>,
}

impl Caches {
    fn build(pil: &PilOut) -> Self {
        let ag_len = pil.air_groups.len();
        let mut symbols_by_air = vec![Vec::new(); ag_len];
        let mut hints_by_air = vec![Vec::new(); ag_len];

        /* ---------------- Symbols ---------------- */
        let all_symbols: &Vec<Symbol> = &pil.symbols;
        for (ag_idx, ag) in pil.air_groups.iter().enumerate() {
            let group_values: Vec<Symbol> = all_symbols
                .iter()
                .filter(|s| {
                    s.air_group_id == Some(ag_idx as u32) && s.r#type == consts::AIRGROUP_VALUE && s.air_id.is_none()
                })
                .cloned()
                .collect();

            let mut per_air = Vec::with_capacity(ag.airs.len());
            for (air_idx, _air) in ag.airs.iter().enumerate() {
                let mut syms: Vec<Symbol> = all_symbols
                    .iter()
                    .filter(|s| s.air_group_id == Some(ag_idx as u32) && s.air_id == Some(air_idx as u32))
                    .cloned()
                    .collect();
                syms.extend(group_values.clone());
                per_air.push(syms);
            }
            symbols_by_air[ag_idx] = per_air;
        }

        /* ---------------- Hints ---------------- */
        let all_hints: &Vec<Hint> = &pil.hints;
        for (ag_idx, ag) in pil.air_groups.iter().enumerate() {
            let mut per_air = Vec::with_capacity(ag.airs.len());
            for (air_idx, _air) in ag.airs.iter().enumerate() {
                let h: Vec<Hint> = all_hints
                    .iter()
                    .filter(|h| {
                        (h.air_group_id.is_none() || h.air_group_id == Some(ag_idx as u32))
                            && (h.air_id.is_none() || h.air_id == Some(air_idx as u32))
                    })
                    .cloned()
                    .collect();
                per_air.push(h);
            }
            hints_by_air[ag_idx] = per_air;
        }

        Self { symbols_by_air, hints_by_air }
    }
}

/* ───────────────────────── main wrapper ────────────────────────────────── */

#[derive(Debug)]
pub struct AirOut {
    pilout_proxy: PilOutProxy,
    caches: Caches,
}

impl AirOut {
    /* ---------- construction ---------- */

    pub fn from_file(filename: &Path) -> Result<Self, Box<dyn Error>> {
        let pilout_proxy = PilOutProxy::new(filename.to_str().ok_or("filename contains non-UTF-8 characters")?)?;
        let caches = Caches::build(&pilout_proxy.pilout);
        let airout = Self { pilout_proxy, caches };
        airout.print_info(); // mimic JS banner
        Ok(airout)
    }

    /* ---------- delegated getters ---------- */

    pub fn pilout(&self) -> &PilOut {
        &self.pilout_proxy.pilout
    }
    pub fn get_airgroup_idx(&self, name: &str) -> Option<usize> {
        self.pilout_proxy.get_airgroup_idx(name)
    }
    pub fn get_air_idx(&self, ag: usize, name: &str) -> Option<usize> {
        self.pilout_proxy.get_air_idx(ag, name)
    }
    pub fn get_air(&self, ag: usize, air: usize) -> &pilout::pilout::Air {
        self.pilout_proxy.get_air(ag, air)
    }
    pub fn find_air(&self, g: &str, a: &str) -> Option<&pilout::pilout::Air> {
        self.pilout_proxy.find_air(g, a)
    }
    pub fn num_stages(&self) -> u32 {
        self.pilout_proxy.num_stages()
    }
    pub fn num_rows(&self, ag: usize, air: usize) -> usize {
        self.pilout_proxy.num_rows(ag, air)
    }
    pub fn print_pilout_info(&self) {
        self.pilout_proxy.print_pilout_info();
    }

    /* ---------- JS-style helpers ---------- */

    pub fn symbols(&self, ag: usize, air: usize) -> &[Symbol] {
        self.caches.symbols_by_air.get(ag).and_then(|v| v.get(air)).map(Vec::as_slice).unwrap_or(&[])
    }
    pub fn hints(&self, ag: usize, air: usize) -> &[Hint] {
        self.caches.hints_by_air.get(ag).and_then(|v| v.get(air)).map(Vec::as_slice).unwrap_or(&[])
    }

    /* ---------- banner (info output) ---------- */

    fn print_info(&self) {
        let pil = &self.pilout_proxy.pilout;

        tracing::info!("[AirOut    ]     Name: {}", pil.name.as_deref().unwrap_or("<unnamed>"));
        tracing::info!("[AirOut    ]     #Airgroups: {}", pil.air_groups.len());
        tracing::info!("[AirOut    ]     #ProofValues: {}", pil.num_proof_values.len());
        tracing::info!("[AirOut    ]     #PublicValues: {}", pil.num_public_values);

        if !pil.public_tables.is_empty() {
            tracing::info!("[AirOut    ]     #PublicTables: {}", pil.public_tables.len());
        }
        if !pil.expressions.is_empty() {
            tracing::info!("[AirOut    ]     #Expressions: {}", pil.expressions.len());
        }
        if !pil.constraints.is_empty() {
            tracing::info!("[AirOut    ]     #Constraints: {}", pil.constraints.len());
        }
        if !pil.hints.is_empty() {
            tracing::info!("[AirOut    ]     #Hints: {}", pil.hints.len());
        }
        if !pil.symbols.is_empty() {
            tracing::info!("[AirOut    ]     #Symbols: {}", pil.symbols.len());
        }

        for (_ag_idx, ag) in pil.air_groups.iter().enumerate() {
            tracing::info!(
                "[AirOut    ]         > Airgroup '{}': ({} airs)",
                ag.name.as_deref().unwrap_or("<unnamed>"),
                ag.airs.len()
            );
            for (_air_idx, air) in ag.airs.iter().enumerate() {
                tracing::info!(
                    "[AirOut    ]             + Air '{}'  rows: {}  stages: {}",
                    air.name.as_deref().unwrap_or("<unnamed>"),
                    air.num_rows.unwrap_or(0),
                    air.stage_widths.len().max(1)
                );
                let exprs = self.pilout().expressions.len();
                let constr = air.constraints.len();
                tracing::info!("[AirOut    ]               Expressions: {}  Constraints: {}", exprs, constr);
            }
        }
    }
}
