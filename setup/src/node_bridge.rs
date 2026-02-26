use std::{
    io::BufReader,
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    thread,
};
use std::io::{BufRead, Write};

use anyhow::Result;

/// Line-delimited JSON messages received from the Node.js wrapper on stdout.
#[derive(serde::Deserialize, Debug)]
#[serde(tag = "type")]
pub(crate) enum NodeMessage {
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

/// Spawn a Node.js process, invoke `fn_name` with `args` over the line-delimited
/// JSON protocol, and return the `value` field from the result message.
///
/// Log messages from the child are forwarded to stderr. Any `error` message
/// or a non-zero exit code is surfaced as an `Err`.
pub(crate) fn call_node(
    js_root: &Path,
    js_file: &str,
    max_old_space_size: Option<u64>,
    fn_name: &str,
    args: serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    let js_file = js_root.join(js_file);
    anyhow::ensure!(js_file.exists(), "Wrapper not found: {}", js_file.display());

    let request = serde_json::json!({
        "type": "call",
        "fn":   fn_name,
        "args": args,
    });
    let mut node_args: Vec<String> = Vec::new();
    if let Some(mb) = max_old_space_size {
        node_args.push(format!("--max-old-space-size={mb}"));
    }
    node_args.push(js_file.to_string_lossy().into_owned());

    let child = Command::new("node")
        .args(&node_args)
        .current_dir(js_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to spawn node: {e}"));

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[ERROR] {e}");
            return Err(e);
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

    stdin.write_all((serde_json::to_string(&request)? + "\n").as_bytes())?;
    stdin.flush()?;
    drop(stdin);

    let result = wait_for_result(&rx);

    let _ = reader_thread.join();

    let status = child.wait()?;
    if !status.success() {
        let msg = format!("Node process exited with {status}");
        eprintln!("[ERROR] {}", msg);
        return Err(anyhow::anyhow!(msg));
    }

    result
}

fn wait_for_result(rx: &mpsc::Receiver<StdoutEvent>) -> Result<Option<serde_json::Value>> {
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
}
