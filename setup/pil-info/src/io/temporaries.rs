//! Temporary allocation for a code block: packs the `tmp` operands of each dimension into as
//! few slots as their lifetimes allow.

use crate::error::{PilInfoError, Result};
use crate::types::code::{CodeOperation, CodeType, OpType};

/// Determines if two lifetime segments overlap (open intervals on the right).
fn is_intersecting(seg1: &[i64; 3], seg2: &[i64; 3]) -> bool {
    seg2[0] < seg1[1] && seg1[0] < seg2[1]
}

/// Packs lifetime segments into non-overlapping subsets using a greedy
/// closest-fit algorithm (matching the JS `temporalsSubsets`).
fn temporals_subsets(segments: &mut [[i64; 3]]) -> Vec<Vec<[i64; 3]>> {
    segments.sort_by_key(|s| s[1]);
    let mut subsets: Vec<Vec<[i64; 3]>> = Vec::new();

    for segment in segments.iter() {
        let mut closest_idx: Option<usize> = None;
        let mut min_distance = i64::MAX;

        for (i, subset) in subsets.iter().enumerate() {
            // Every subset starts with a segment.
            let Some(last) = subset.last() else { continue };
            if is_intersecting(segment, last) {
                continue;
            }
            let distance = (last[1] - segment[0]).abs();
            if distance < min_distance {
                min_distance = distance;
                closest_idx = Some(i);
            }
        }

        if let Some(idx) = closest_idx {
            subsets[idx].push(*segment);
        } else {
            subsets.push(vec![*segment]);
        }
    }
    subsets
}

/// Analyses tmp variable lifetimes and assigns compacted IDs: `id1d` for the base-field tmps and
/// `id3d` for those in the extension of dimension `ext_dim` (none when `ext_dim` is 1).
///
/// Fails on a tmp of `code` numbered `maxid` or above, or of a dimension neither 1 nor `ext_dim`.
pub fn get_id_maps(
    maxid: usize,
    id1d: &mut [i64],
    id3d: &mut [i64],
    code: &[CodeOperation],
    ext_dim: u64,
) -> Result<(u64, u64)> {
    let mut ini1d = vec![-1i64; maxid];
    let mut end1d = vec![-1i64; maxid];
    let mut ini3d = vec![-1i64; maxid];
    let mut end3d = vec![-1i64; maxid];

    for (j, r) in code.iter().enumerate() {
        let op = j;
        let j = j as i64;
        // The dest, then the sources.
        for t in std::iter::once(&r.dest).chain(&r.src) {
            if t.op_type != OpType::Tmp {
                continue;
            }
            let id = tmp_id(t, op, maxid)?;
            let (ini, end) = if t.dim == 1 {
                (&mut ini1d, &mut end1d)
            } else if t.dim == ext_dim {
                (&mut ini3d, &mut end3d)
            } else {
                return Err(PilInfoError::TmpDimension { op, id, dim: t.dim, ext_dim });
            };
            if ini[id] == -1 {
                ini[id] = j;
            }
            end[id] = j;
        }
    }

    let mut segments1d: Vec<[i64; 3]> = Vec::new();
    let mut segments3d: Vec<[i64; 3]> = Vec::new();
    for j in 0..maxid {
        if ini1d[j] >= 0 {
            segments1d.push([ini1d[j], end1d[j], j as i64]);
        }
        if ini3d[j] >= 0 {
            segments3d.push([ini3d[j], end3d[j], j as i64]);
        }
    }

    let subsets1d = temporals_subsets(&mut segments1d);
    let subsets3d = temporals_subsets(&mut segments3d);

    let mut count1d: u64 = 0;
    for s in &subsets1d {
        for a in s {
            id1d[a[2] as usize] = count1d as i64;
        }
        count1d += 1;
    }
    let mut count3d: u64 = 0;
    for s in &subsets3d {
        for a in s {
            id3d[a[2] as usize] = count3d as i64;
        }
        count3d += 1;
    }
    Ok((count1d, count3d))
}

/// The id of tmp `t` of op `op`, which must be below `maxid`.
fn tmp_id(t: &CodeType, op: usize, maxid: usize) -> Result<usize> {
    match usize::try_from(t.id) {
        Ok(id) if id < maxid => Ok(id),
        _ => Err(PilInfoError::TmpOutOfRange { op, id: t.id, max_id: maxid }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporals_subsets_empty() {
        let mut segments: Vec<[i64; 3]> = Vec::new();
        let result = temporals_subsets(&mut segments);
        assert!(result.is_empty());
    }

    #[test]
    fn test_temporals_subsets_non_overlapping() {
        let mut segments = vec![[0, 2, 0], [3, 5, 1], [6, 8, 2]];
        let result = temporals_subsets(&mut segments);
        // All fit in one subset since none overlap
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_temporals_subsets_overlapping() {
        let mut segments = vec![[0, 3, 0], [1, 4, 1], [5, 7, 2]];
        let result = temporals_subsets(&mut segments);
        // First two overlap, third fits in first subset
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_is_intersecting() {
        assert!(is_intersecting(&[0, 3, 0], &[1, 4, 1]));
        assert!(!is_intersecting(&[0, 2, 0], &[2, 4, 1]));
        assert!(!is_intersecting(&[0, 2, 0], &[3, 5, 1]));
    }

    fn tmp(id: u64, dim: u64) -> CodeType {
        CodeType { op_type: OpType::Tmp, id, dim, ..Default::default() }
    }

    fn copy(dest: CodeType, src: CodeType) -> CodeOperation {
        CodeOperation { op: "copy".to_string(), dest, src: vec![src] }
    }

    #[test]
    fn temporaries_of_both_dimensions_get_slots() {
        let code = vec![copy(tmp(0, 1), tmp(1, 3)), copy(tmp(2, 1), tmp(0, 1))];
        let (mut id1d, mut id3d) = (vec![-1; 3], vec![-1; 3]);
        assert_eq!(get_id_maps(3, &mut id1d, &mut id3d, &code, 3).unwrap(), (1, 1));
        assert_eq!((id1d, id3d), (vec![0, -1, 0], vec![-1, 0, -1]));
    }

    /// Were the `assert!`s of `get_id_maps`.
    #[test]
    fn a_temporary_out_of_range_or_of_another_dimension_is_an_error() {
        let (mut id1d, mut id3d) = (vec![-1; 2], vec![-1; 2]);
        let err = get_id_maps(2, &mut id1d, &mut id3d, &[copy(tmp(0, 1), tmp(2, 1))], 3).unwrap_err();
        assert!(matches!(err, PilInfoError::TmpOutOfRange { op: 0, id: 2, max_id: 2 }), "{err}");

        let err = get_id_maps(2, &mut id1d, &mut id3d, &[copy(tmp(0, 1), tmp(1, 3))], 1).unwrap_err();
        assert!(matches!(err, PilInfoError::TmpDimension { op: 0, id: 1, dim: 3, ext_dim: 1 }), "{err}");
    }
}
