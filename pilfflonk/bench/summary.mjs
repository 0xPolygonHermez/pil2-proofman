// The summary of the pilfflonk benchmark (bench.sh summary,
// pilfflonk/docs/performance.md#reproducing): for every point (program, size, layout, threads) of
// $BENCH_DIR/{compile,setup,prove}.tsv, the median of its runs and their spread, as markdown tables.
//
//     node pilfflonk/bench/summary.mjs <BENCH_DIR>
//
// The spread is (max − min)/median of the runs of a point, in percent; the load is the range of
// the 1-minute load averages the runs started with.

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { argv, exit, stderr } from "node:process";

if (argv.length !== 3) {
    stderr.write("usage: summary.mjs <BENCH_DIR>\n");
    exit(2);
}
const dir = argv[2];

function read(name) {
    const path = join(dir, name);
    if (!existsSync(path)) return [];
    const [head, ...lines] = readFileSync(path, "utf8").trim().split("\n");
    const columns = head.split("\t");
    return lines.filter((l) => l.length > 0).map((l) => Object.fromEntries(l.split("\t").map((v, i) => [columns[i], v])));
}

function median(values) {
    const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
    if (v.length === 0) return NaN;
    const m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function stat(rows, key) {
    const v = rows.map((r) => Number(r[key])).filter((x) => Number.isFinite(x));
    const m = median(v);
    const spread = v.length > 1 && m > 0 ? (100 * (Math.max(...v) - Math.min(...v))) / m : 0;
    return { m, spread, n: v.length };
}

const fmt = (x, digits = 2) => (Number.isFinite(x) ? x.toFixed(digits) : "–");
const gb = (kb) => (Number.isFinite(kb) ? (kb / 1024 / 1024).toFixed(2) : "–");
const loadRange = (rows) => {
    const v = rows.map((r) => Number(r.load1));
    return `${Math.min(...v).toFixed(0)}–${Math.max(...v).toFixed(0)}`;
};

function groups(rows, keys) {
    const out = new Map();
    for (const r of rows) {
        const k = keys.map((key) => r[key]).join("\t");
        if (!out.has(k)) out.set(k, []);
        out.get(k).push(r);
    }
    return [...out.values()].sort((a, b) => {
        for (const key of keys) {
            const x = a[0][key], y = b[0][key];
            const c = Number.isFinite(Number(x)) ? Number(x) - Number(y) : x.localeCompare(y);
            if (c !== 0) return c;
        }
        return 0;
    });
}

function table(title, head, lines) {
    if (lines.length === 0) return;
    console.log(`\n### ${title}\n`);
    console.log(`| ${head.join(" | ")} |`);
    console.log(`|${head.map(() => "---").join("|")}|`);
    for (const l of lines) console.log(`| ${l.join(" | ")} |`);
}

const compile = read("compile.tsv");
table(
    "pil2com",
    ["program", "N", "runs", "load", "s", "GB", "pilout MB"],
    groups(compile, ["program", "bits"]).map((g) => [
        g[0].program,
        `2^${g[0].bits}`,
        g.length,
        loadRange(g),
        fmt(stat(g, "wall_s").m, 1),
        gb(stat(g, "rss_kb").m),
        fmt(stat(g, "pilout_bytes").m / 1e6, 1),
    ]),
);

const setup = read("setup.tsv").filter((r) => r.exit === "0");
table(
    "setup-pilfflonk",
    ["program", "N", "layout", "threads", "runs", "load", "s (spread %)", "GB", "nBitsExt", "qDeg", "f", "SRS powers", ".const MB", "SRS MB"],
    groups(setup, ["program", "bits", "packing", "threads"]).map((g) => {
        const t = stat(g, "wall_s");
        return [
            g[0].program,
            `2^${g[0].bits}`,
            g[0].packing,
            g[0].threads,
            g.length,
            loadRange(g),
            `${fmt(t.m)} (${fmt(t.spread, 0)})`,
            gb(stat(g, "rss_kb").m),
            g[0].nBitsExt,
            g[0].qDeg,
            g[0].nF,
            g[0].srs_powers,
            fmt(Number(g[0].const_bytes) / 1e6, 0),
            fmt(Number(g[0].srs_bytes) / 1e6, 0),
        ];
    }),
);

const prove = read("prove.tsv").filter((r) => r.exit === "0" && r.verify_exit === "0");
const points = groups(prove, ["program", "bits", "packing", "threads"]);
table(
    "pilfflonk prove and verify",
    ["program", "N", "layout", "threads", "runs", "load", "prove s (spread %)", "GB", "verify s", "proof bytes"],
    points.map((g) => {
        const t = stat(g, "wall_s");
        return [
            g[0].program,
            `2^${g[0].bits}`,
            g[0].packing,
            g[0].threads,
            g.length,
            loadRange(g),
            `${fmt(t.m)} (${fmt(t.spread, 0)})`,
            gb(stat(g, "rss_kb").m),
            fmt(stat(g, "verify_s").m),
            g[0].proof_bytes,
        ];
    }),
);

// The phases, in seconds (the C++ timers): the key's loading (the SRS, the fixed columns and their
// commitments), the witness (read and made an instance), each stage (hints, im pols, INTT, MSM), Q
// (coset extension, domain and evaluation, interpolation, MSM, and the rest: its buffers, the check
// of its bound, its pieces), the evaluations at ξ, the opening (W and W', [W] and [W'], and the
// rest: the interpolants r_i), and what no timer covers (the key's JSON files and the vkey's
// digest, the transcript, the proof's files, the process's exit).
const m = (g, k) => (Number.isFinite(stat(g, k).m) ? stat(g, k).m : 0);
const sum = (g, keys) => keys.reduce((s, k) => s + m(g, k), 0);
table(
    "Phases of the prover (median s)",
    ["program", "N", "layout", "threads", "key", "witness", "hints", "im pols", "INTT", "MSM", "Q ext.", "Q eval.", "Q interp.", "Q MSM", "Q rest", "evals", "W, W'", "[W], [W']", "open rest", "rest", "total"],
    points.map((g) => {
        const q = ["q_extend", "q_domain", "q_evaluate", "q_interpolate", "q_commit"];
        const open = ["shplonk_w", "shplonk_wp", "shplonk_commit_w", "shplonk_commit_wp"];
        const cells = [
            sum(g, ["load_srs", "load_airs", "fixed_commitments"]),
            m(g, "witness"),
            sum(g, ["s1_hints", "s2_hints"]),
            sum(g, ["s1_im", "s2_im"]),
            sum(g, ["s1_intt", "s2_intt"]),
            sum(g, ["s1_msm", "s2_msm"]),
            m(g, "q_extend"),
            sum(g, ["q_domain", "q_evaluate"]),
            m(g, "q_interpolate"),
            m(g, "q_commit"),
            Math.max(0, m(g, "q_total") - sum(g, q)),
            m(g, "evaluations"),
            sum(g, ["shplonk_w", "shplonk_wp"]),
            sum(g, ["shplonk_commit_w", "shplonk_commit_wp"]),
            Math.max(0, m(g, "open") - sum(g, open)),
        ];
        const total = stat(g, "wall_s").m;
        // Run by run, the wall time less the top-level timers, and then its median: the median of
        // each column need not add up to the total.
        const top = ["load_srs", "load_airs", "fixed_commitments", "witness", "s1_total", "s2_total", "q_total", "evaluations", "open"];
        const value = (r, k) => (Number.isFinite(Number(r[k])) ? Number(r[k]) : 0);
        const rest = median(g.map((r) => Number(r.wall_s) - top.reduce((s, k) => s + value(r, k), 0)));
        return [g[0].program, `2^${g[0].bits}`, g[0].packing, g[0].threads, ...cells.map((c) => fmt(c)), fmt(rest), fmt(total)];
    }),
);

table(
    "Peak RSS of each phase of the prover (median GB)",
    ["program", "N", "layout", "threads", "key", "witness", "stage 1", "stage 2", "Q ext.", "Q eval.", "Q interp.", "Q MSM", "evals", "open", "peak"],
    points.map((g) => [
        g[0].program,
        `2^${g[0].bits}`,
        g[0].packing,
        g[0].threads,
        ...["rss_load", "rss_witness", "rss_s1", "rss_s2", "rss_q_extend", "rss_q_evaluate", "rss_q_interpolate", "rss_q_commit", "rss_evaluations", "rss_open"].map((k) =>
            gb(stat(g, k).m),
        ),
        gb(stat(g, "rss_kb").m),
    ]),
);
