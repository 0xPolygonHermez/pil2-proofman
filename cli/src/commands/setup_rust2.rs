use clap::Parser;
use colored::Colorize;
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

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
    /// ok=true may carry an optional value (Part 1 result)
    #[serde(rename = "result")]
    Result {
        ok: bool,
        #[serde(default)]
        value: Option<serde_json::Value>,
    },
    #[serde(rename = "error")]
    Error { message: String, stack: Option<String> },
}

/// Events emitted by the stdout reader thread.
enum StdoutEvent {
    Msg(NodeMessage),
    Raw(String), // non-JSON line — print verbatim
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
        let pilout   = abs(&self.pilout)?;
        let builddir = abs(Path::new(&self.builddir))?;
        let binfiles: Vec<PathBuf> = self.binfiles.iter().map(|p| abs(p)).collect::<Result<_, _>>()?;
        let std_path = self.std_path.as_deref().map(abs).transpose()?;
        let fixed    = self.fixed.as_deref().map(abs).transpose()?;

        let starkstructs: serde_json::Value = match &self.starkstructs {
            Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)?,
            None => json!({}),
        };

        // Config object that setup_cmd.js expects.
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

        let builddir_str = builddir.to_string_lossy().to_string();

        // Build node command-line arguments.
        let mut node_args: Vec<String> = Vec::new();
        if let Some(mb) = self.max_old_space_size {
            node_args.push(format!("--max-old-space-size={mb}"));
        }
        node_args.push(wrapper.to_string_lossy().into_owned());

        // Spawn node. stdin/stdout are piped; stderr flows directly to the terminal
        // so the user sees all JS console output in real time.
        let mut child = Command::new("node")
            .args(&node_args)
            .current_dir(&self.js_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn node: {e}"))?;

        let mut stdin = child.stdin.take().expect("stdin not captured");
        let stdout    = child.stdout.take().expect("stdout not captured");

        // Spawn a dedicated reader thread so stdout is always drained.
        // This prevents the pipe buffer from filling up when Part 1 returns
        // large starkInfo / verifierInfo objects, which would deadlock if we
        // tried to write Part 2 while Node is blocked writing Part 1 output.
        let (tx, rx) = mpsc::channel::<StdoutEvent>();
        let reader_thread = thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) if l.trim().is_empty() => {}
                    Ok(l) => {
                        let event = match serde_json::from_str::<NodeMessage>(&l) {
                            Ok(msg) => StdoutEvent::Msg(msg),
                            Err(_)  => StdoutEvent::Raw(l),
                        };
                        if tx.send(event).is_err() { break; }
                    }
                    Err(_) => break,
                }
            }
        });

        // -----------------------------------------------------------------
        // Helper: drain the channel until a Result or Error message arrives.
        // Returns the optional value payload (present in Part 1 result).
        // -----------------------------------------------------------------
        let wait_result = |rx: &mpsc::Receiver<StdoutEvent>|
            -> Result<Option<serde_json::Value>, Box<dyn std::error::Error + Send + Sync>>
        {
            loop {
                match rx.recv()? {
                    StdoutEvent::Msg(NodeMessage::Result { ok: true, value }) => return Ok(value),
                    StdoutEvent::Msg(NodeMessage::Result { ok: false, .. }) => {
                        return Err("Node returned ok=false".into());
                    }
                    StdoutEvent::Msg(NodeMessage::Error { message, stack }) => {
                        eprintln!("[ERROR] {}", message);
                        if let Some(s) = stack { eprintln!("{}", s); }
                        return Err(message.into());
                    }
                    StdoutEvent::Msg(NodeMessage::Log { level, msg }) => match level.as_str() {
                        "error" => eprintln!("[ERROR] {}", msg),
                        "warn"  => eprintln!("[WARN]  {}", msg),
                        _       => eprintln!("[INFO]  {}", msg),
                    },
                    StdoutEvent::Raw(line) => println!("{}", line),
                }
            }
        };

        // -----------------------------------------------------------------
        // Phase 1: call setup_part1 — runs starkSetup for every air
        // -----------------------------------------------------------------
        let part1_req = json!({
            "type": "call",
            "fn":   "setup_part1",
            "id":   1,
            "args": { "config": &config, "buildDir": &builddir_str }
        });
        stdin.write_all((serde_json::to_string(&part1_req)? + "\n").as_bytes())?;
        stdin.flush()?;

        let setup_data = wait_result(&rx)?
            .ok_or("setup_part1 returned no value")?;

        // Rust holds the setup data. Currently it just passes it through to
        // Part 2; in the future this is where Rust will compute it instead.

        // -----------------------------------------------------------------
        // Phase 2: call setup_part2 — file I/O, const tree, binfiles, aggregation
        // -----------------------------------------------------------------
        let part2_req = json!({
            "type": "call",
            "fn":   "setup_part2",
            "id":   2,
            "args": { "config": &config, "buildDir": &builddir_str, "starkStructs": setup_data }
        });
        stdin.write_all((serde_json::to_string(&part2_req)? + "\n").as_bytes())?;
        stdin.flush()?;
        drop(stdin); // close stdin → EOF signals Node that no more requests are coming

        wait_result(&rx)?;

        // Wait for the reader thread and child process to finish.
        let _ = reader_thread.join();

        let status = child.wait()?;
        if !status.success() {
            let msg = format!("Node process exited with {status}");
            eprintln!("[ERROR] {}", msg);
            return Err(msg.into());
        }

        println!("Setup completed successfully");
        Ok(())
    }
}
