//! BLAKE3 message-permutation connection generator.
//!
//! Produces the two `CONN` fixed columns for the *offline* connection (copy) argument
//! that `blake3.pil` uses to enforce the SIGMA message schedule.
//!
//! One can run the binary e.g.:
//!     cargo run --bin blake3-connection -- --bits 21 build -o blake3_connection_fixed.bin
//!     cargo run --bin blake3-connection -- --bits 21 validate
//!     cargo run --bin blake3-connection -- --bits 21 cell x 8     # debug a single cell
//!     cargo run --bin blake3-connection -- --bits 21 row 8
//!     cargo run --bin blake3-connection -- --bits 21 frame 0
//!     cargo run --bin blake3-connection -- --bits 21 cycle x 1
//!     cargo run --bin blake3-connection -- --bits 21 range y 0 16

use fields::{Field, Goldilocks, PrimeField64};
use proofman_common::{write_fixed_cols_bin, FixedColsInfo};

type F = Goldilocks;

// ════════════════════════════════════════════════════════════════════════════════
// Constants (lifted from pil2-components/lib/std/pil/goldilocks.pil — same values the
// PIL `connection()` uses on the assumes side, so the two sides cancel on identity rows)
// ════════════════════════════════════════════════════════════════════════════════

/// GEN[i] generates the multiplicative subgroup of order 2^i. For N = 2^bits, g = GEN[bits].
const GEN: [u64; 33] = [
    1,
    18446744069414584320,
    281474976710656,
    18446744069397807105,
    17293822564807737345,
    70368744161280,
    549755813888,
    17870292113338400769,
    13797081185216407910,
    1803076106186727246,
    11353340290879379826,
    455906449640507599,
    17492915097719143606,
    1532612707718625687,
    16207902636198568418,
    17776499369601055404,
    6115771955107415310,
    12380578893860276750,
    9306717745644682924,
    18146160046829613826,
    3511170319078647661,
    17654865857378133588,
    5416168637041100469,
    16905767614792059275,
    9713644485405565297,
    5456943929260765144,
    17096174751763063430,
    1213594585890690845,
    6414415596519834757,
    16116352524544190054,
    9123114210336311365,
    4614640910117430873,
    1753635133440165772,
];

/// Coset separator used by the connection argument: column i uses k_coset^i.
const K_COSET: u64 = 12275445934081160404;

// ─── BLAKE3 layout (must match blake3.pil) ───────────────────────────────────────
const CLOCKS_PER_G: usize = 1;
const NUM_G_FUNCTIONS: usize = 8;
const ROUNDS: usize = 7;
const ROWS_PER_ROUND: usize = NUM_G_FUNCTIONS * CLOCKS_PER_G; // 8
const CLOCKS: usize = ROWS_PER_ROUND * ROUNDS; // 56

/// Message word permutation schedule. SIGMA[r][s] = original message index used at round r, slot s.
const SIGMA: [[usize; 16]; ROUNDS] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

// ════════════════════════════════════════════════════════════════════════════════
// Connection builder
// ════════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct Decoded {
    frame: usize,
    round: usize,
    slot: usize,
    word: usize, // m[word] stored here = SIGMA[round][slot]
}

struct Blake3Connection {
    bits: usize,
    n: usize,
    num_blake3: usize,
    active_rows: usize,               // num_blake3 * CLOCKS ; rows beyond are identity
    g: F,                             // GEN[bits]
    k: F,                             // K_COSET
    sigma_inv: [[usize; 16]; ROUNDS], // sigma_inv[r][word] = slot s with SIGMA[r][s] == word
}

impl Blake3Connection {
    fn new(bits: usize) -> Self {
        assert!((1..=32).contains(&bits), "bits must be in 1..=32");
        let n = 1usize << bits;

        // NUM_BLAKE3 — identical formula to blake3.pil.
        let nnu = n % CLOCKS;
        let num_blake3 = if nnu == 0 {
            assert!(n >= CLOCKS, "N too small to fit one Blake3");
            n / CLOCKS
        } else {
            assert!(n >= 2 * CLOCKS, "N too small to fit one Blake3");
            (n - nnu) / CLOCKS - 1
        };

        let mut sigma_inv = [[0usize; 16]; ROUNDS];
        for r in 0..ROUNDS {
            for s in 0..16 {
                sigma_inv[r][SIGMA[r][s]] = s;
            }
        }

        Self {
            bits,
            n,
            num_blake3,
            active_rows: num_blake3 * CLOCKS,
            g: F::from_u64(GEN[bits]),
            k: F::from_u64(K_COSET),
            sigma_inv,
        }
    }

    // ── pure cell math (no allocation — safe to query any cell even at N = 2^21) ──

    /// g^row (the identity column value at `row`).
    fn id(&self, row: usize) -> F {
        self.g.exp_u64(row as u64)
    }

    /// Connection-group base value of a cell: k_coset^col * g^row.
    fn base(&self, col: usize, row: usize) -> F {
        let idr = self.id(row);
        if col == 0 {
            idr
        } else {
            self.k * idr
        }
    }

    /// Decode a cell into (frame, round, slot, word), or `None` if it is not an active
    /// message cell (tail rows / rows beyond the last Blake3 instance).
    fn decode(&self, col: usize, row: usize) -> Option<Decoded> {
        if row >= self.active_rows {
            return None;
        }
        let frame = row / CLOCKS;
        let wf = row % CLOCKS;
        let round = wf / ROWS_PER_ROUND;
        let row_off = wf % ROWS_PER_ROUND;
        let slot = 2 * row_off + col; // col 0 -> even slot, col 1 -> odd slot
        Some(Decoded { frame, round, slot, word: SIGMA[round][slot] })
    }

    /// The next cell in the connection cycle (same word, next round). Identity for non-message cells.
    fn next_cell(&self, col: usize, row: usize) -> (usize, usize) {
        match self.decode(col, row) {
            None => (col, row), // fixed point
            Some(d) => {
                let nr = (d.round + 1) % ROUNDS;
                let ns = self.sigma_inv[nr][d.word];
                let ncol = ns % 2;
                let nrow = d.frame * CLOCKS + nr * ROWS_PER_ROUND + ns / 2;
                (ncol, nrow)
            }
        }
    }

    /// CONN fixed-column value at (col, row) = base value of the next cell in the cycle.
    fn conn(&self, col: usize, row: usize) -> F {
        let (nc, nr) = self.next_cell(col, row);
        self.base(nc, nr)
    }

    /// The full connection cycle containing (col, row), starting at (col, row).
    fn cycle(&self, col: usize, row: usize) -> Vec<(usize, usize)> {
        let start = (col, row);
        let mut cyc = vec![start];
        let mut cur = self.next_cell(col, row);
        while cur != start && cyc.len() <= ROUNDS + 1 {
            cyc.push(cur);
            cur = self.next_cell(cur.0, cur.1);
        }
        cyc
    }

    // ── bulk build ──

    /// Build both CONN columns as `Vec<F>` (length N each). CONN[0] -> x_packed, CONN[1] -> y_packed.
    /// O(N): ID via running product, then overwrite each active message cell with its cycle pointer.
    fn build_columns(&self) -> [Vec<F>; 2] {
        let mut id = vec![F::ZERO; self.n];
        let mut acc = F::ONE;
        for v in id.iter_mut() {
            *v = acc;
            acc *= self.g;
        }

        let mut cx = id.clone();
        let mut cy: Vec<F> = id.iter().map(|&v| self.k * v).collect();

        let base_with = |col: usize, row: usize| -> F {
            if col == 0 {
                id[row]
            } else {
                self.k * id[row]
            }
        };
        for f in 0..self.num_blake3 {
            let frame = f * CLOCKS;
            for r in 0..ROUNDS {
                let nr = (r + 1) % ROUNDS;
                for s in 0..16 {
                    let word = SIGMA[r][s];
                    let ns = self.sigma_inv[nr][word];
                    let col = s % 2;
                    let row = frame + r * ROWS_PER_ROUND + s / 2;
                    let ncol = ns % 2;
                    let nrow = frame + nr * ROWS_PER_ROUND + ns / 2;
                    let v = base_with(ncol, nrow);
                    if col == 0 {
                        cx[row] = v
                    } else {
                        cy[row] = v
                    }
                }
            }
        }
        [cx, cy]
    }

    /// Write the two CONN columns to `out_file` in the format the PIL loads via
    /// `#pragma extern_fixed_file`. `air_name` must match the alias used at instantiation.
    fn write_bin(&self, airgroup_name: &str, air_name: &str, out_file: &str) {
        let [cx, cy] = self.build_columns();
        let conn0 = FixedColsInfo::new(&format!("{air_name}.CONN"), Some(vec![0]), cx);
        let conn1 = FixedColsInfo::new(&format!("{air_name}.CONN"), Some(vec![1]), cy);
        write_fixed_cols_bin(out_file, airgroup_name, air_name, self.n as u64, &mut [conn0, conn1]);
        println!(
            "wrote CONN[0], CONN[1] ({} rows, {} Blake3 instances) to {} as {}.{}",
            self.n, self.num_blake3, out_file, air_name, "CONN"
        );
    }

    // ── validation (index-based: cheap even at N = 2^21) ──

    fn validate(&self) {
        let idx = |col: usize, row: usize| col * self.n + row;
        let total = 2 * self.n;

        // 1) bijection: cell -> next_cell hits every cell exactly once.
        let mut hit = vec![false; total];
        for col in 0..2 {
            for row in 0..self.n {
                let (nc, nr) = self.next_cell(col, row);
                let t = idx(nc, nr);
                assert!(!hit[t], "not a bijection: cell {:?} targeted twice", (nc, nr));
                hit[t] = true;
            }
        }

        // 2) cycle structure: non-trivial cycles all have length ROUNDS, count == frames*16.
        let mut visited = vec![false; total];
        let mut nontrivial = 0usize;
        for col in 0..2 {
            for row in 0..self.n {
                if visited[idx(col, row)] {
                    continue;
                }
                let mut len = 0usize;
                let (mut c, mut r) = (col, row);
                while !visited[idx(c, r)] {
                    visited[idx(c, r)] = true;
                    len += 1;
                    let (nc, nr) = self.next_cell(c, r);
                    c = nc;
                    r = nr;
                }
                if len > 1 {
                    nontrivial += 1;
                    assert_eq!(len, ROUNDS, "cycle at {:?} has length {}", (col, row), len);
                }
            }
        }
        assert_eq!(nontrivial, self.num_blake3 * 16, "unexpected cycle count");
        println!(
            "validate: OK — bits={} N={} NUM_BLAKE3={} : valid permutation, {} cycles of length {}",
            self.bits, self.n, self.num_blake3, nontrivial, ROUNDS
        );
    }

    // ── pretty printers ──

    fn col_name(col: usize) -> char {
        if col == 0 {
            'x'
        } else {
            'y'
        }
    }

    fn dump_cell(&self, col: usize, row: usize) {
        let name = Self::col_name(col);
        let (nc, nr) = self.next_cell(col, row);
        println!("cell {}[row={}]   (CONN column index {})", name, row, col);
        match self.decode(col, row) {
            Some(d) => println!(
                "  schedule       frame {}, round {}, slot {}  -> holds message word m[{}]",
                d.frame, d.round, d.slot, d.word
            ),
            None => println!("  schedule       inactive cell (tail / beyond last Blake3) -> identity"),
        }
        println!("  g = GEN[{}]   = {}", self.bits, GEN[self.bits]);
        println!("  ID(row)=g^{}  = {}", row, self.id(row).as_canonical_u64());
        println!("  base(this)     = {}", self.base(col, row).as_canonical_u64());
        println!("  CONN(this)     = {}   (= base of next cell)", self.conn(col, row).as_canonical_u64());
        println!("  next cell ->   {}[row={}]", Self::col_name(nc), nr);
        let pretty: Vec<String> =
            self.cycle(col, row).iter().map(|&(c, r)| format!("{}[{}]", Self::col_name(c), r)).collect();
        println!("  cycle (len {}) {}", pretty.len(), pretty.join(" -> "));
    }

    fn dump_row(&self, row: usize) {
        println!("── row {} ──", row);
        for col in 0..2 {
            self.dump_cell(col, row);
        }
    }

    fn dump_frame(&self, f: usize) {
        assert!(f < self.num_blake3, "frame {} out of range (NUM_BLAKE3={})", f, self.num_blake3);
        let frame = f * CLOCKS;
        println!("── frame {} (rows {}..{}) ──", f, frame, frame + CLOCKS);
        println!("  round  row    x=m[..]  y=m[..]");
        for r in 0..ROUNDS {
            for ro in 0..ROWS_PER_ROUND {
                let row = frame + r * ROWS_PER_ROUND + ro;
                println!("   {:>2}   {:>5}    m[{:>2}]    m[{:>2}]", r, row, SIGMA[r][2 * ro], SIGMA[r][2 * ro + 1]);
            }
        }
    }

    fn dump_range(&self, col: usize, start: usize, end: usize) {
        let name = Self::col_name(col);
        let end = end.min(self.n);
        println!("── CONN[{}] rows {}..{} ──", name, start, end);
        for row in start..end {
            let (nc, nr) = self.next_cell(col, row);
            println!(
                "  {}[{:>6}]  CONN={:>20}   -> {}[{}]",
                name,
                row,
                self.conn(col, row).as_canonical_u64(),
                Self::col_name(nc),
                nr
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// CLI
// ════════════════════════════════════════════════════════════════════════════════

// Defaults for `build`. Adjust to match your PIL instantiation / desired output path.
const DEFAULT_AIRGROUP: &str = "Hashes";
const DEFAULT_AIR: &str = "Blake3";
const DEFAULT_OUT: &str = "src/blake3_connection_fixed.bin";

const BITS: usize = 20;

fn parse_col(s: &str) -> usize {
    match s {
        "x" | "X" | "0" => 0,
        "y" | "Y" | "1" => 1,
        _ => panic!("col must be x|y|0|1, got {s:?}"),
    }
}

fn flag<'a>(args: &'a [String], name: &str, default: &'a str) -> &'a str {
    args.iter().position(|a| a == name).map(|i| args[i + 1].as_str()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut bits = BITS;
    let mut rest = &args[1..];
    if rest.len() >= 2 && rest[0] == "--bits" {
        bits = rest[1].parse().expect("bad --bits value");
        rest = &rest[2..];
    }

    let c = Blake3Connection::new(bits);

    if rest.is_empty() {
        println!("Blake3 connection generator — subcommands: build|validate|cell|row|frame|cycle|range");
        c.validate();
        println!();
        c.dump_cell(0, 8); // x[8]: round 1, slot 0, word m[2]
        return;
    }

    match rest[0].as_str() {
        "build" => {
            let out = flag(rest, "-o", flag(rest, "--output", DEFAULT_OUT));
            let airgroup = flag(rest, "--airgroup", DEFAULT_AIRGROUP);
            let air = flag(rest, "--air", DEFAULT_AIR);
            c.write_bin(airgroup, air, out);
        }
        "validate" => c.validate(),
        "cell" => c.dump_cell(parse_col(&rest[1]), rest[2].parse().unwrap()),
        "row" => c.dump_row(rest[1].parse().unwrap()),
        "frame" => c.dump_frame(rest[1].parse().unwrap()),
        "cycle" => {
            let pretty: Vec<String> = c
                .cycle(parse_col(&rest[1]), rest[2].parse().unwrap())
                .iter()
                .map(|&(col, r)| format!("{}[{}]", Blake3Connection::col_name(col), r))
                .collect();
            println!("cycle (len {}): {}", pretty.len(), pretty.join(" -> "));
        }
        "range" => c.dump_range(parse_col(&rest[1]), rest[2].parse().unwrap(), rest[3].parse().unwrap()),
        other => panic!("unknown subcommand {other:?}"),
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_orders() {
        for i in 1..=32 {
            let g = F::from_u64(GEN[i]);
            assert_eq!(g.exp_u64(1u64 << i), F::ONE, "GEN[{i}] order");
            assert_ne!(g.exp_u64(1u64 << (i - 1)), F::ONE, "GEN[{i}] not primitive");
        }
    }

    #[test]
    fn permutation_and_cycles_small() {
        for bits in [7, 8, 9, 10] {
            Blake3Connection::new(bits).validate();
        }
    }

    #[test]
    fn build_matches_lazy() {
        let c = Blake3Connection::new(8);
        let [cx, cy] = c.build_columns();
        for row in 0..c.n {
            assert_eq!(cx[row], c.conn(0, row), "cx mismatch at {row}");
            assert_eq!(cy[row], c.conn(1, row), "cy mismatch at {row}");
        }
    }

    #[test]
    fn conn_is_permutation_of_base() {
        // CONN must be a permutation of the base multiset {g^row} ∪ {k·g^row}.
        let c = Blake3Connection::new(8);
        let [cx, cy] = c.build_columns();
        let mut got: Vec<u64> = cx.iter().chain(cy.iter()).map(|v| v.as_canonical_u64()).collect();
        let mut want: Vec<u64> = (0..c.n)
            .map(|r| c.base(0, r).as_canonical_u64())
            .chain((0..c.n).map(|r| c.base(1, r).as_canonical_u64()))
            .collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn known_cell_x8() {
        // x[8]: frame 0, round 1, slot 0, word m[2]; next is round 2's m[2] = y[18].
        let c = Blake3Connection::new(8);
        let d = c.decode(0, 8).unwrap();
        assert_eq!((d.frame, d.round, d.slot, d.word), (0, 1, 0, 2));
        assert_eq!(c.next_cell(0, 8), (1, 18));
        assert_eq!(c.next_cell(0, 1), (0, 8)); // round-0 m[2] = x[1] -> x[8]
    }
}
