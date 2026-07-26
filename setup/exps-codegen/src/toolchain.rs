//! CUDA toolchain glue: arch resolution, the nvcc include/define flag set, and
//! compile/link invocations. Same arch convention as the pil2-stark build
//! (`auto` / `major` / explicit list); the objects the autotuner compiles are
//! exactly what the final `.so` links.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAJOR_ARCHS: &[u32] = &[80, 86, 89, 90, 100, 120];

pub struct Toolchain {
    pub arch_list: Vec<u32>,
    /// `-gencode …` flags (one pair per arch + forward-PTX), used by both compile and link.
    gencode: Vec<String>,
    /// Compile flags after `nvcc -c`: -Xcompiler/-std/-O3 + gencode + defines + includes.
    compile_flags: Vec<String>,
}

/// Resolve the requested arch spec into a sorted, de-duplicated arch list.
/// `auto` (or empty) detects the host GPU and falls back to `major`.
fn resolve_archs(archspec: &str) -> Vec<u32> {
    let spec = archspec.trim().to_lowercase();
    if spec == "major" {
        return MAJOR_ARCHS.to_vec();
    }
    if spec.is_empty() || spec == "auto" {
        // __nvcc_device_query prints the host compute capability (e.g. "120").
        if let Ok(out) = Command::new("__nvcc_device_query").output() {
            if out.status.success() {
                let digits: String = String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(a) = digits.parse::<u32>() {
                    eprintln!("[exps-codegen] auto-detected host GPU arch: sm_{a}");
                    return vec![a];
                }
            }
        }
        eprintln!("[exps-codegen] arch auto-detect failed -> building 'major': {MAJOR_ARCHS:?}");
        return MAJOR_ARCHS.to_vec();
    }
    // explicit list: "89,120" / "sm_120" / "120"
    let mut archs: Vec<u32> = spec.replace("sm_", "").split(',').filter_map(|s| s.trim().parse::<u32>().ok()).collect();
    archs.sort_unstable();
    archs.dedup();
    archs
}

impl Toolchain {
    pub fn new(stark_src: Option<PathBuf>, archspec: &str, work_dir: &Path) -> Result<Self> {
        let stark_dir = match stark_src {
            Some(p) => p,
            None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pil2-stark"),
        };
        let stark_dir = stark_dir.canonicalize().unwrap_or(stark_dir);
        if !stark_dir.join("src").is_dir() {
            bail!("pil2-stark source not found at {} (pass --stark-src)", stark_dir.display());
        }

        let arch_list = resolve_archs(archspec);
        if arch_list.is_empty() {
            bail!("no valid CUDA archs in spec '{archspec}'");
        }

        // gencode: SASS for every arch + PTX for the newest arch of each Blackwell
        // lineage (sm_100-11x datacenter vs the rest) for forward compatibility.
        let mut gencode: Vec<String> = Vec::new();
        for a in &arch_list {
            gencode.push("-gencode".into());
            gencode.push(format!("arch=compute_{a},code=sm_{a}"));
        }
        // Blackwell datacenter lineage = arch whose decimal starts with 10 or 11 (e.g. 100, 110).
        let is_dc = |a: &u32| {
            let s = a.to_string();
            s.starts_with("10") || s.starts_with("11")
        };
        let ptx_dc = arch_list.iter().filter(|a| is_dc(a)).max().copied();
        let ptx_rest = arch_list.iter().filter(|a| !is_dc(a)).max().copied();
        for p in [ptx_dc, ptx_rest].into_iter().flatten() {
            gencode.push("-gencode".into());
            gencode.push(format!("arch=compute_{p},code=compute_{p}"));
        }

        let src = stark_dir.join("src");
        let s = |rel: &str| format!("-I{}", src.join(rel).display());
        let k = |rel: &str| format!("-I{}", stark_dir.join(rel).display());
        let mut incs: Vec<String> = vec![
            format!("-I{}", work_dir.display()),
            k("external/sppark/ff"),
            k("external/sppark"),
            k("external/sppark/util"),
            k("external/sppark/ec"),
            k("external/blst/src"),
            format!("-I{}", src.display()),
            s("utils"),
            s("goldilocks"),
            s("goldilocks/utils"),
            s("goldilocks/src"),
            s("binfile"),
            s("XKCP"),
            s("bctree"),
            s("config"),
            s("api"),
            s("bn128"),
            s("bn128/src"),
            s("bn128/src/ffiasm"),
            s("bn128/src/poseidon"),
            s("bn128/src/msm"),
            s("bn128/src/ntt"),
            s("bn128/src/poseidon2"),
            s("bn128/src/curve"),
            s("starkpil"),
            s("starkpil/expressions"),
            s("starkpil/transcript"),
            s("starkpil/merkleTree"),
            s("starkpil/fri"),
            s("rapidsnark"),
            s("rapidsnark/polynomial"),
            s("fflonk_setup"),
            "-I/usr/include".into(),
            "-I/usr/local/include".into(),
            "-I/usr/lib/x86_64-linux-gnu/openmpi/include".into(),
        ];

        let mut defs: Vec<String> = [
            "-D__USE_CUDA__",
            "-DGL64_PARTIALLY_REDUCED",
            "-D__AVX2__",
            "-D__USE_ASSEMBLY__",
            "-D__ADX__",
            "-DFEATURE_BN254",
            "-DUSE_CUDA_GRAPH",
            "-DOMPI_SKIP_MPICXX",
            "-DMPICH_SKIP_MPICXX",
            "-D__USE_MPI_RMA__",
            "--diag-suppress",
            "114",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        // The generated TU includes goldilocks_trace_layout.cuh, so any resolveLayout() it
        // compiles must make the SAME decision as the prover's libstarksgpu.a. Without this the
        // kernel resolves ColMajor while the LDE writes ColMajorTiled, and proofs are silently
        // wrong. Mirrors cm_layout()/store_qq() in emit.rs and -DFORCE_TILED_LAYOUT in the Makefile.
        if std::env::var("FORCE_TILED_LAYOUT").map(|v| v == "1").unwrap_or(false) {
            defs.push("-DFORCE_TILED_LAYOUT".to_string());
        }

        let mut compile_flags: Vec<String> = ["-Xcompiler", "-fPIC", "-Xcompiler", "-mavx2", "-std=c++17", "-O3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        compile_flags.extend(gencode.iter().cloned());
        compile_flags.extend(defs);
        compile_flags.append(&mut incs);

        let _ = &stark_dir; // validated above; not retained
        Ok(Toolchain { arch_list, gencode, compile_flags })
    }

    pub fn arch_summary(&self) -> String {
        self.arch_list.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(" ")
    }

    /// Compile one TU to an object. `extra_inc` (the autotuner probe dir) is
    /// searched in addition to the standard includes. Returns (ok, stderr).
    pub fn compile_tu(&self, cuf: &Path, obj: &Path, extra_inc: Option<&Path>) -> Result<(bool, String)> {
        let mut cmd = Command::new("nvcc");
        cmd.arg("-c").args(&self.compile_flags).arg(cuf).arg("-o").arg(obj);
        if let Some(inc) = extra_inc {
            cmd.arg(format!("-I{}", inc.display()));
        }
        let out = cmd.output().context("failed to spawn nvcc")?;
        Ok((out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned()))
    }

    /// Link pre-compiled objects into a shared library (uses only gencode).
    pub fn link_objs(&self, objs: &[PathBuf], dest: &Path) -> Result<()> {
        let out = Command::new("nvcc")
            .arg("-shared")
            .args(&self.gencode)
            .args(objs)
            .args(LINK_FLAGS)
            .arg(dest)
            .output()
            .context("failed to spawn nvcc (link)")?;
        if !out.status.success() {
            bail!("link {} failed:\n{}", dest.display(), String::from_utf8_lossy(&out.stderr));
        }
        Ok(())
    }

    /// Compile sources and link them into a shared library in one nvcc call
    /// (the no-autotune fallback, where no objects were kept).
    pub fn compile_link_cus(&self, cus: &[PathBuf], dest: &Path) -> Result<()> {
        let out = Command::new("nvcc")
            .arg("-shared")
            .args(&self.compile_flags)
            .args(cus)
            .args(LINK_FLAGS)
            .arg(dest)
            .output()
            .context("failed to spawn nvcc (compile+link)")?;
        if !out.status.success() {
            bail!("compile+link {} failed:\n{}", dest.display(), String::from_utf8_lossy(&out.stderr));
        }
        Ok(())
    }
}

/// Trailing link flags shared by both link paths, ending in `-o` so the caller
/// appends the destination. `-cudart static` embeds the CUDA runtime so the
/// `.so` carries no `libcudart.so.<major>` dependency and loads regardless of
/// the host's CUDA toolkit major, fixing `dlopen failed: libcudart.so.N`
/// errors. `--strip-all` offsets the ~700KB the static runtime adds.
const LINK_FLAGS: &[&str] = &["-cudart", "static", "-Xlinker", "--strip-all", "-o"];
