//! Tracing-subscriber setup and per-rank log formatting for MPI runs.

use crate::{RankInfo, VerboseMode};
use colored::Colorize;
use proofman_starks_lib_c::set_log_level_c;
use std::io::IsTerminal;
use std::sync::OnceLock;
use tracing::dispatcher;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FormatFields;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::time::SystemTime;
use tracing_subscriber::fmt::FormatEvent;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use yansi::Color;
use yansi::Paint;

static GLOBAL_RANK: OnceLock<i32> = OnceLock::new();
static VERBOSE_MODE: OnceLock<VerboseMode> = OnceLock::new();
static IS_TERMINAL: OnceLock<bool> = OnceLock::new();

pub struct RankFormatter;

impl<S, N> FormatEvent<S, N> for RankFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &fmt::FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let timer = SystemTime;

        let mut time_str = String::new();
        {
            let mut fake_writer = Writer::new(&mut time_str);
            timer.format_time(&mut fake_writer)?;
        }

        let is_terminal = IS_TERMINAL.get().copied().unwrap_or(false);

        if is_terminal {
            write!(writer, "{} ", time_str.dimmed())?;
        } else {
            write!(writer, "{time_str} ")?;
        }

        if let Some(rank) = GLOBAL_RANK.get().copied() {
            let rank_str = match is_terminal {
                true => format!("[rank={rank}]").dimmed(),
                false => format!("[rank={rank}]").into(),
            };
            write!(writer, "{rank_str} ")?;
        }

        let target = event.metadata().target();
        let show_target =
            VERBOSE_MODE.get().map(|vm| matches!(vm, VerboseMode::Debug | VerboseMode::Trace)).unwrap_or(false);

        if is_terminal {
            if show_target {
                write!(writer, "{} ", target.dimmed())?;
            }

            let level_str = match *event.metadata().level() {
                tracing::Level::TRACE => "TRACE".paint(Color::Cyan),
                tracing::Level::DEBUG => "DEBUG".paint(Color::Blue),
                tracing::Level::INFO => "INFO".paint(Color::Green),
                tracing::Level::WARN => "WARN".paint(Color::Yellow),
                tracing::Level::ERROR => "ERROR".paint(Color::Red),
            };
            write!(writer, "{level_str}: ")?;
        } else {
            let level_str = match *event.metadata().level() {
                tracing::Level::TRACE => "TRACE",
                tracing::Level::DEBUG => "DEBUG",
                tracing::Level::INFO => "INFO",
                tracing::Level::WARN => "WARN",
                tracing::Level::ERROR => "ERROR",
            };
            if show_target {
                write!(writer, "{target} ")?;
            }
            write!(writer, "{level_str}: ")?;
        }

        let mut visitor = MessageVisitor::new();
        event.record(&mut visitor);
        write!(writer, "{}", visitor.message)?;
        writeln!(writer)?;

        Ok(())
    }
}

struct MessageVisitor {
    message: String,
}

impl MessageVisitor {
    fn new() -> Self {
        Self { message: String::new() }
    }
}

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
            if self.message.starts_with('"') && self.message.ends_with('"') {
                self.message = self.message[1..self.message.len() - 1].to_string();
            }
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }
}

pub fn set_global_rank(rank: i32) {
    let _ = GLOBAL_RANK.set(rank);
}

pub fn initialize_logger(verbose_mode: VerboseMode, rank: Option<&RankInfo>) {
    if GLOBAL_RANK.get().is_none() {
        if let Some(r) = rank {
            if r.n_processes > 1 {
                set_global_rank(r.world_rank);
            }
        }
    }

    let _ = VERBOSE_MODE.set(verbose_mode);

    let is_terminal = std::io::stdout().is_terminal() && std::env::var("NO_COLOR").is_err();

    let _ = IS_TERMINAL.set(is_terminal);

    // Disable ANSI/colors globally when not in a terminal
    if !is_terminal {
        yansi::disable();
    }

    if dispatcher::has_been_set() {
        return;
    }

    let stdout_layer = tracing_subscriber::fmt::layer()
        .event_format(RankFormatter)
        .with_writer(std::io::stdout)
        .with_ansi(is_terminal)
        .with_filter(LevelFilter::from(verbose_mode));

    tracing_subscriber::registry().with(stdout_layer).init();

    set_log_level_c(verbose_mode.into());
}
