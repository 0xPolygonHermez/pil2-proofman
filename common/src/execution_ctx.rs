use std::path::PathBuf;
use crate::{DistributionCtx, StdMode, VerboseMode};
use std::sync::RwLock;
use std::collections::HashMap;
#[allow(dead_code)]
/// Represents the context when executing a witness computer plugin
pub struct ExecutionCtx {
    pub rom_path: Option<PathBuf>,
    pub cached_buffers_path: Option<HashMap<String, PathBuf>>,
    /// If true, the plugin must generate the public outputs
    pub public_output: bool,
    pub verbose_mode: VerboseMode,
    pub dctx: RwLock<DistributionCtx>,
    pub std_mode: StdMode,
}

impl ExecutionCtx {
    pub fn builder() -> ExecutionCtxBuilder {
        ExecutionCtxBuilder::new()
    }
}

pub struct ExecutionCtxBuilder {
    rom_path: Option<PathBuf>,
    cached_buffers_path: Option<HashMap<String, PathBuf>>,
    public_output: bool,
    verbose_mode: VerboseMode,
    std_mode: StdMode,
}

impl Default for ExecutionCtxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionCtxBuilder {
    pub fn new() -> Self {
        ExecutionCtxBuilder {
            rom_path: None,
            cached_buffers_path: None,
            public_output: true,
            verbose_mode: VerboseMode::Info,
            std_mode: StdMode::default(),
        }
    }

    pub fn with_rom_path(mut self, rom_path: Option<PathBuf>) -> Self {
        self.rom_path = rom_path;
        self
    }

    pub fn with_cached_buffers_path(mut self, cached_buffers_path: Option<HashMap<String, PathBuf>>) -> Self {
        self.cached_buffers_path = cached_buffers_path;
        self
    }

    pub fn with_verbose_mode(mut self, verbose_mode: VerboseMode) -> Self {
        self.verbose_mode = verbose_mode;
        self
    }

    pub fn with_std_mode(mut self, std_mode: StdMode) -> Self {
        self.std_mode = std_mode;
        self
    }

    pub fn build(self) -> ExecutionCtx {
        ExecutionCtx {
            rom_path: self.rom_path,
            cached_buffers_path: self.cached_buffers_path,
            public_output: self.public_output,
            verbose_mode: self.verbose_mode,
            dctx: RwLock::new(DistributionCtx::new()),
            std_mode: self.std_mode,
        }
    }
}
