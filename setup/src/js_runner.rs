use std::{
    io::BufReader,
    process::{Command, Stdio},
    sync::mpsc,
    thread,
};
use proofman_common::StarkStruct;
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, Write};

use anyhow::Result;

use crate::setup::SetupConfig;

/// Line-delimited JSON messages received from the Node.js wrapper on stdout.
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum NodeMessage {
    #[serde(rename = "log")]
    Log { level: String, msg: String },
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

pub(crate) fn run_part2_js(
    config: &SetupConfig,
    starkstructs: serde_json::Value,
    stark_structs: Vec<Vec<StarkStruct>>,
) -> std::result::Result<(), anyhow::Error> {
    let wrapper = config.js_root.join("src/cmd/setup_cmd_wrapper.js");

    anyhow::ensure!(wrapper.exists(), "Wrapper not found: {}", wrapper.display());

    let node_config = json!({
        "airout": {
            "airoutFilename": config.pilout.to_string_lossy()
        },
        "setup": {
            "settings":            &starkstructs,
            "genAggregationSetup": config.recursive,
            "optImPols":           config.impols,
            "binFiles":            config.binfiles.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
            "stdPath":             config.std_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            "fixedPath":           config.fixed.as_ref().map(|p| p.to_string_lossy().to_string()),
        }
    });
    let builddir_str = config.builddir.to_string_lossy().to_string();

    let stark_structs_json = serde_json::to_value(&stark_structs)?;

    let mut node_args: Vec<String> = Vec::new();
    if let Some(mb) = config.max_old_space_size {
        node_args.push(format!("--max-old-space-size={mb}"));
    }
    node_args.push(wrapper.to_string_lossy().into_owned());
    let child = Command::new("node")
        .args(&node_args)
        .current_dir(&config.js_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("Failed to spawn node: {e}"));
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[ERROR] {e}");
            return Err(anyhow::anyhow!(e));
        }
    };

    let mut stdin = child.stdin.take().expect("stdin not captured");
    let stdout = child.stdout.take().expect("stdout not captured");
    let (tx, rx) = mpsc::channel::<StdoutEvent>();
    let reader_thread = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(l) if l.trim().is_empty() => {}
                Ok(l) => {
                    let event = match serde_json::from_str::<NodeMessage>(&l) {
                        Ok(msg) => StdoutEvent::Msg(msg),
                        Err(_) => StdoutEvent::Raw(l),
                    };
                    if tx.send(event).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let wait_result = |rx: &mpsc::Receiver<StdoutEvent>| -> Result<Option<serde_json::Value>> {
        loop {
            match rx.recv()? {
                StdoutEvent::Msg(NodeMessage::Result { ok: true, value }) => return Ok(value),
                StdoutEvent::Msg(NodeMessage::Result { ok: false, .. }) => {
                    return Err(anyhow::anyhow!("Node returned ok=false"));
                }
                StdoutEvent::Msg(NodeMessage::Error { message, stack }) => {
                    eprintln!("[ERROR] {}", message);
                    if let Some(s) = stack {
                        eprintln!("{}", s);
                    }
                    return Err(anyhow::anyhow!(message));
                }
                StdoutEvent::Msg(NodeMessage::Log { level, msg }) => match level.as_str() {
                    "error" => eprintln!("[ERROR] {}", msg),
                    "warn" => eprintln!("[WARN]  {}", msg),
                    _ => eprintln!("[INFO]  {}", msg),
                },
                StdoutEvent::Raw(line) => println!("{}", line),
            }
        }
    };
    let part2_req = json!({
        "type": "call",
        "fn":   "setup_part2",
        "id":   2,
        "args": { "config": &node_config, "buildDir": &builddir_str, "starkStructs": stark_structs_json }
    });
    stdin.write_all((serde_json::to_string(&part2_req)? + "\n").as_bytes())?;
    stdin.flush()?;
    drop(stdin);
    wait_result(&rx)?;
    let _ = reader_thread.join();
    let status = child.wait()?;
    Ok(if !status.success() {
        let msg = format!("Node process exited with {status}");
        eprintln!("[ERROR] {}", msg);
        return Err(anyhow::anyhow!(msg));
    })
}
