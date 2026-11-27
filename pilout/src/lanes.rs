//! Lanes: columns of inlined template instances (`Symbol.lane`), laid out as the proto says.

use std::collections::BTreeMap;

use crate::pilout::{Air, Symbol, SymbolType};

/// A column of a lane: its name in the template and lane 0's `Symbol.name`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneField {
    pub name: String,
    pub symbol: String,
}

/// The lanes of one `laneName`: `count` is `None` for a single inline, `start` lane 0's column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneGroup {
    pub name: String,
    pub count: Option<usize>,
    pub start: usize,
    pub fields: Vec<LaneField>,
}

fn size(s: &Symbol) -> usize {
    s.lengths.iter().map(|&l| l as usize).product()
}

/// The lane groups of an air, in column order; an air without lanes is not checked.
pub fn air_lanes<'a>(air: &Air, air_symbols: impl IntoIterator<Item = &'a Symbol>) -> Result<Vec<LaneGroup>, String> {
    let mut stage1: Vec<&Symbol> =
        air_symbols.into_iter().filter(|s| s.r#type == SymbolType::WitnessCol as i32 && s.stage == Some(1)).collect();
    if stage1.iter().all(|s| s.lane.is_none()) {
        return Ok(vec![]);
    }
    stage1.sort_by_key(|s| s.id);

    let mut next = 0;
    for s in &stage1 {
        if s.id as usize != next {
            return Err(format!("symbol `{}` is at column {}, not {next}", s.name, s.id));
        }
        next += size(s);
    }
    let width = air.stage_widths.first().copied().unwrap_or(0) as usize;
    if next != width {
        return Err(format!("its stage-1 symbols fill {next} columns, not {width}"));
    }

    let mut by_group: BTreeMap<&str, BTreeMap<Option<u32>, Vec<&Symbol>>> = BTreeMap::new();
    for s in &stage1 {
        if let Some(lane) = &s.lane {
            by_group.entry(&lane.lane_name).or_default().entry(lane.lane_index).or_default().push(s);
        }
    }

    let mut groups = vec![];
    for (name, lanes) in by_group {
        let count = match lanes.keys().next() {
            Some(None) if lanes.len() == 1 => None,
            Some(None) => return Err(format!("lane `{name}` mixes a single inline with an array")),
            _ if lanes.keys().enumerate().any(|(i, k)| *k != Some(i as u32)) => {
                return Err(format!(
                    "lane `{name}` has indices {:?}, not 0..{}",
                    lanes.keys().flatten().collect::<Vec<_>>(),
                    lanes.len()
                ))
            }
            _ => Some(lanes.len()),
        };

        let shape = |lane: &[&Symbol]| -> Vec<(String, Vec<u32>)> {
            lane.iter().map(|s| (s.lane.as_ref().unwrap().name.clone(), s.lengths.clone())).collect()
        };
        let lanes: Vec<Vec<&Symbol>> = lanes.into_values().collect();
        let mut names: Vec<_> = shape(&lanes[0]).into_iter().map(|(n, _)| n).collect();
        names.sort();
        if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
            return Err(format!("lane `{name}` has column `{}` twice", w[0]));
        }
        let start = lanes[0][0].id as usize;
        let lane_width: usize = lanes[0].iter().map(|s| size(s)).sum();
        for (l, lane) in lanes.iter().enumerate() {
            if shape(lane) != shape(&lanes[0]) {
                return Err(format!("lane {l} of `{name}` does not hold the columns of lane 0"));
            }
            // Columns already tile, so a lane at its place with no gaps is contiguous.
            if lane[0].id as usize != start + l * lane_width
                || lane.windows(2).any(|w| w[1].id as usize != w[0].id as usize + size(w[0]))
            {
                return Err(format!("lane {l} of `{name}` is not the block at column {}", start + l * lane_width));
            }
        }
        let fields = lanes[0]
            .iter()
            .map(|s| LaneField { name: s.lane.as_ref().unwrap().name.clone(), symbol: s.name.clone() })
            .collect();
        groups.push(LaneGroup { name: name.to_string(), count, start, fields });
    }
    groups.sort_by_key(|g| g.start);
    Ok(groups)
}

/// `s`'s PIL name: `Air.x.col` or `Air.x[i].col` for a lane's column, else `Symbol.name`.
pub fn pil_name(s: &Symbol) -> String {
    let Some(lane) = &s.lane else {
        return s.name.clone();
    };
    let prefix = s.name.split_once('.').map(|(air, _)| format!("{air}.")).unwrap_or_default();
    match lane.lane_index {
        Some(i) => format!("{prefix}{}[{i}].{}", lane.lane_name, lane.name),
        None => format!("{prefix}{}.{}", lane.lane_name, lane.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pilout::LaneSymbol;

    fn witness(name: &str, id: u32, lengths: &[u32], lane: Option<(&str, &str, Option<u32>)>) -> Symbol {
        Symbol {
            name: name.into(),
            r#type: SymbolType::WitnessCol as i32,
            id,
            stage: Some(1),
            dim: lengths.len() as u32,
            lengths: lengths.to_vec(),
            lane: lane.map(|(lane_name, name, lane_index)| LaneSymbol {
                lane_name: lane_name.into(),
                name: name.into(),
                lane_index,
            }),
            ..Default::default()
        }
    }

    fn air(stage1: u32) -> Air {
        Air { stage_widths: vec![stage1], ..Default::default() }
    }

    /// Columns: solo 0 | bin 1..4 | binhi[0] 4..7 | binhi[1] 7..10, each lane `pepe[2]`, `pepa`.
    fn symbols() -> Vec<Symbol> {
        vec![
            witness("X.binhi__1__pepa", 9, &[], Some(("binhi", "pepa", Some(1)))),
            witness("X.solo", 0, &[], None),
            witness("X.bin__pepe", 1, &[2], Some(("bin", "pepe", None))),
            witness("X.bin__pepa", 3, &[], Some(("bin", "pepa", None))),
            witness("X.binhi__0__pepe", 4, &[2], Some(("binhi", "pepe", Some(0)))),
            witness("X.binhi__0__pepa", 6, &[], Some(("binhi", "pepa", Some(0)))),
            witness("X.binhi__1__pepe", 7, &[2], Some(("binhi", "pepe", Some(1)))),
        ]
    }

    #[test]
    fn groups_follow_the_columns() {
        let fields = |prefix: &str| {
            ["pepe", "pepa"].map(|f| LaneField { name: f.into(), symbol: format!("X.{prefix}{f}") }).to_vec()
        };
        let bin = LaneGroup { name: "bin".into(), count: None, start: 1, fields: fields("bin__") };
        let hi = LaneGroup { name: "binhi".into(), count: Some(2), start: 4, fields: fields("binhi__0__") };
        assert_eq!(air_lanes(&air(10), &symbols()), Ok(vec![bin, hi]));
    }

    #[test]
    fn an_air_without_lanes_has_none() {
        // Unchecked, as before lanes.
        assert_eq!(air_lanes(&air(9), &[witness("a", 0, &[2], None), witness("b", 0, &[], None)]), Ok(vec![]));
    }

    #[test]
    fn a_lane_of_one_is_an_array() {
        let s = [witness("X.solo", 0, &[], None), witness("X.h", 1, &[], Some(("h", "a", Some(0))))];
        assert_eq!(air_lanes(&air(2), &s).unwrap()[0].count, Some(1));
    }

    #[test]
    fn a_broken_layout_is_refused() {
        let with = |f: &dyn Fn(&mut Vec<Symbol>)| {
            let mut s = symbols();
            f(&mut s);
            air_lanes(&air(10), &s)
        };
        assert!(air_lanes(&air(11), &symbols()).is_err());
        // Lane 1 holding another column.
        assert!(with(&|s| s[0].lane.as_mut().unwrap().name = "pepo".into()).is_err());
        // Lanes swapped: lane 1 before lane 0.
        assert!(with(&|s| {
            for x in s.iter_mut().filter(|x| x.name.starts_with("X.binhi")) {
                let lane = x.lane.as_mut().unwrap();
                lane.lane_index = lane.lane_index.map(|i| 1 - i);
            }
        })
        .is_err());
        // A missing index.
        assert!(with(&|s| {
            for x in s.iter_mut().filter(|x| x.name.starts_with("X.binhi__1")) {
                x.lane.as_mut().unwrap().lane_index = Some(2);
            }
        })
        .is_err());
        // A column twice in a lane.
        assert!(with(&|s| s[2].lane.as_mut().unwrap().name = "pepa".into()).is_err());
        // A single inline mixed with an array.
        assert!(with(&|s| s[0].lane.as_mut().unwrap().lane_index = None).is_err());
    }

    #[test]
    fn pil_names_rebuild_the_inline() {
        let s = symbols();
        assert_eq!(pil_name(&s[0]), "X.binhi[1].pepa");
        assert_eq!(pil_name(&s[1]), "X.solo");
        assert_eq!(pil_name(&s[2]), "X.bin.pepe");
    }
}
