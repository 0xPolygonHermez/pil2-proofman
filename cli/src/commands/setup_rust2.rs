use clap::Parser;
use colored::Colorize;
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Resolve a path to absolute using the Rust process's cwd as base.
/// Unlike `canonicalize`, this works even if the path doesn't exist yet (e.g. builddir).
fn abs(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

/// Line-delimited JSON messages received from the Node.js wrapper on stdout.
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum NodeMessage {
    #[serde(rename = "log")]
    Log { level: String, msg: String },
    #[serde(rename = "result")]
    Result { ok: bool },
    #[serde(rename = "error")]
    Error { message: String, stack: Option<String> },
}

#[derive(Parser)]
#[command(version, about = "Run setup via Node.js backend (incremental Rust port)", long_about = None)]
#[command(propagate_version = true)]
pub struct SetupRust2Cmd {
    /// Path to the pilout .ptb file
    #[clap(short = 'a', long = "pilout")]
    pilout: PathBuf,

    /// Build output directory
    #[clap(short = 'b', long = "builddir", default_value = "tmp")]
    builddir: String,

    /// Binary files (repeatable: -i file1 -i file2)
    #[clap(short = 'i', long = "binfiles", num_args = 0..)]
    binfiles: Vec<PathBuf>,

    /// Stark structs JSON file
    #[clap(short = 's', long = "starkstructs")]
    starkstructs: Option<PathBuf>,

    /// Standard path
    #[clap(short = 't', long = "std-path")]
    std_path: Option<PathBuf>,

    /// Generate aggregation/recursive setup
    #[clap(short = 'r', long = "recursive")]
    recursive: bool,

    /// Optimize intermediate polynomials
    #[clap(short = 'm', long = "impols")]
    impols: bool,

    /// Fixed polynomials path
    #[clap(short = 'u', long = "fixed")]
    fixed: Option<PathBuf>,

    /// Path to the pil2-proofman-js root directory
    #[clap(long = "js-root")]
    js_root: PathBuf,

    /// Node.js max old space size in MB (e.g. 65536)
    #[clap(long = "max-old-space-size")]
    max_old_space_size: Option<u64>,
}

impl SetupRust2Cmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} SetupRust2", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let wrapper = self.js_root.join("src/cmd/setup_cmd_wrapper.js");
        if !wrapper.exists() {
            return Err(format!("Wrapper not found: {}", wrapper.display()).into());
        }

        // Resolve all paths to absolute so they work regardless of Node's cwd.
        let pilout    = abs(&self.pilout)?;
        let builddir  = abs(Path::new(&self.builddir))?;
        let binfiles: Vec<PathBuf> = self.binfiles.iter().map(|p| abs(p)).collect::<Result<_, _>>()?;
        let std_path  = self.std_path.as_deref().map(abs).transpose()?;
        let fixed     = self.fixed.as_deref().map(abs).transpose()?;

        let starkstructs: serde_json::Value = match &self.starkstructs {
            Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)?,
            None => json!({}),
        };

        let config = json!({
            "airout": {
                "airoutFilename": pilout.to_string_lossy()
            },
            "setup": {
                "settings":            starkstructs,
                "genAggregationSetup": self.recursive,
                "optImPols":           self.impols,
                "binFiles":            binfiles.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
                "stdPath":             std_path.as_ref().map(|p| p.to_string_lossy().to_string()),
                "fixedPath":           fixed.as_ref().map(|p| p.to_string_lossy().to_string()),
            }
        });

        let request = json!({
            "type": "call",
            "fn":   "setup",
            "args": {
                "config":   config,
                "buildDir": builddir.to_string_lossy()
            }
        });

        // Spawn `node src/cmd/setup_cmd_wrapper.js` inside js-root so that
        // relative require() paths resolve correctly.
        let mut node_args: Vec<String> = Vec::new();
        if let Some(mb) = self.max_old_space_size {
            node_args.push(format!("--max-old-space-size={mb}"));
        }
        node_args.push(wrapper.to_string_lossy().into_owned());

        let mut child = Command::new("node")
            .args(&node_args)
            .current_dir(&self.js_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // JS console output flows directly to terminal
            .spawn()
            .map_err(|e| format!("Failed to spawn node: {e}"))?;

        // Send the request and close stdin (signals EOF to Node).
        {
            let mut stdin = child.stdin.take().expect("stdin not captured");
            stdin.write_all(serde_json::to_string(&request)?.as_bytes())?;
        } // drop → EOF

        // Process structured JSON messages from stdout.
        let stdout = child.stdout.take().expect("stdout not captured");
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<NodeMessage>(&line) {
                Ok(NodeMessage::Log { level, msg }) => match level.as_str() {
                    "error" => eprintln!("[ERROR] {}", msg),
                    "warn"  => eprintln!("[WARN]  {}", msg),
                    _       => eprintln!("[INFO]  {}", msg),
                },
                Ok(NodeMessage::Result { ok: true }) => {
                    println!("Setup completed successfully");
                }
                Ok(NodeMessage::Result { ok: false }) => {
                    return Err("Node setup returned ok=false".into());
                }
                Ok(NodeMessage::Error { message, stack }) => {
                    eprintln!("[ERROR] {}", message);
                    if let Some(s) = &stack {
                        eprintln!("{}", s);
                    }
                    return Err(message.into());
                }
                Err(_) => {
                    // Not a protocol message — print verbatim (raw Node output).
                    println!("{}", line);
                }
            }
        }

        let status = child.wait()?;
        if !status.success() {
            let msg = format!("Node process exited with {status}");
            eprintln!("[ERROR] {}", msg);
            return Err(msg.into());
        }

        Ok(())
    }
}
