//! Temporary allocation for a code block: packs the `tmp` operands of each dimension into as
//! few slots as their lifetimes allow.

use crate::types::code::{CodeOperation, OpType};

const FIELD_EXTENSION: u64 = 3;

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
            let last = subset.last().unwrap();
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

/// Analyses tmp variable lifetimes and assigns compacted IDs.
pub fn get_id_maps(maxid: usize, id1d: &mut [i64], id3d: &mut [i64], code: &[CodeOperation]) -> (u64, u64) {
    let mut ini1d = vec![-1i64; maxid];
    let mut end1d = vec![-1i64; maxid];
    let mut ini3d = vec![-1i64; maxid];
    let mut end3d = vec![-1i64; maxid];

    for (j, r) in code.iter().enumerate() {
        let j = j as i64;
        // Check dest
        if r.dest.op_type == OpType::Tmp {
            let id = r.dest.id as usize;
            assert!(id < maxid, "Id exceeds maxid");
            if r.dest.dim == 1 {
                if ini1d[id] == -1 {
                    ini1d[id] = j;
                    end1d[id] = j;
                } else {
                    end1d[id] = j;
                }
            } else {
                assert_eq!(r.dest.dim, FIELD_EXTENSION);
                if ini3d[id] == -1 {
                    ini3d[id] = j;
                    end3d[id] = j;
                } else {
                    end3d[id] = j;
                }
            }
        }
        // Check sources
        for src in &r.src {
            if src.op_type == OpType::Tmp {
                let id = src.id as usize;
                assert!(id < maxid, "Id exceeds maxid");
                if src.dim == 1 {
                    if ini1d[id] == -1 {
                        ini1d[id] = j;
                        end1d[id] = j;
                    } else {
                        end1d[id] = j;
                    }
                } else {
                    assert_eq!(src.dim, FIELD_EXTENSION);
                    if ini3d[id] == -1 {
                        ini3d[id] = j;
                        end3d[id] = j;
                    } else {
                        end3d[id] = j;
                    }
                }
            }
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
    (count1d, count3d)
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
}
