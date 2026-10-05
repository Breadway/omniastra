use burn::config::Config;

/// Architecture and input-regime configuration.
#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 128)]
    pub d_model: usize,
    /// Relational Transformer layers over (registers + state + history).
    #[config(default = 4)]
    pub n_layers: usize,
    /// Action-query decoder layers.
    #[config(default = 2)]
    pub n_dec_layers: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    /// SwiGLU hidden width.
    #[config(default = 344)]
    pub d_ff: usize,
    /// Learned register (global workspace) tokens.
    #[config(default = 8)]
    pub n_reg: usize,
    /// Size of the game-embedding table (game ids are dataset-local indices).
    #[config(default = 64)]
    pub n_games: usize,
    #[config(default = 300)]
    pub cat_vocab: usize,
    #[config(default = 48)]
    pub sem_vocab: usize,

    // ---- input regime / ablation switches -----------------------------------
    /// Embed game-local categorical ids (template / action-def / zone ids).
    #[config(default = true)]
    pub use_ids: bool,
    /// Use position-indexed numeric slot weights ("slot id" adapter).
    #[config(default = true)]
    pub use_slot_pos: bool,
    /// Use rule-derived descriptors (slot / template / action semantics).
    #[config(default = true)]
    pub use_desc: bool,
    /// Add a per-game embedding to the register tokens.
    #[config(default = false)]
    pub use_game_emb: bool,

    // ---- architecture ablations ---------------------------------------------
    /// Relation-aware attention bias.
    #[config(default = true)]
    pub use_rel: bool,
    /// Relative temporal bias among history tokens.
    #[config(default = true)]
    pub use_time_bias: bool,
    /// Include history tokens (when false they are masked out).
    #[config(default = true)]
    pub use_history: bool,
    /// Registers (when false they are masked out and the value head pools tokens).
    #[config(default = true)]
    pub use_registers: bool,
    /// Action-to-action self-attention in the decoder.
    #[config(default = true)]
    pub action_self_attn: bool,
    /// Add the pointer-style link features (source/target embeddings) to action queries.
    #[config(default = true)]
    pub use_action_links: bool,
}

impl ModelConfig {
    /// <1M parameters: CPU-feasible pilots (a few minutes of training).
    pub fn nano() -> ModelConfig {
        ModelConfig::new().with_d_model(64).with_n_layers(2).with_n_dec_layers(1).with_n_heads(4).with_d_ff(172).with_n_reg(4)
    }
    /// ~1-3M parameters: tests and fast CPU experiments.
    pub fn tiny() -> ModelConfig {
        ModelConfig::new().with_d_model(128).with_n_layers(4).with_n_dec_layers(2).with_n_heads(4).with_d_ff(344).with_n_reg(6)
    }
    /// ~5-20M parameters: initial transfer experiments.
    pub fn small() -> ModelConfig {
        ModelConfig::new().with_d_model(256).with_n_layers(6).with_n_dec_layers(3).with_n_heads(8).with_d_ff(688).with_n_reg(8)
    }
    /// ~20-50M parameters: serious multi-game experiments.
    pub fn medium() -> ModelConfig {
        ModelConfig::new().with_d_model(512).with_n_layers(8).with_n_dec_layers(4).with_n_heads(8).with_d_ff(1376).with_n_reg(16)
    }
    pub fn by_name(name: &str) -> Option<ModelConfig> {
        match name {
            "nano" => Some(Self::nano()),
            "tiny" => Some(Self::tiny()),
            "small" => Some(Self::small()),
            "medium" => Some(Self::medium()),
            _ => None,
        }
    }
}
