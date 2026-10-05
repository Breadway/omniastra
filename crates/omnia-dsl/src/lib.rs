//! OmniDSL: canonical game-definition IR, validation, static analysis,
//! canonical serialisation and hashing.

pub mod analysis;
pub mod ir;
pub mod relabel;
pub mod validate;

pub use analysis::{analyze, Desc, GameDescriptors, DESC_DIM};
pub use ir::*;
pub use relabel::Relabel;
pub use validate::validate;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DslError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("validation failed:\n{0}")]
    Invalid(String),
}

impl GameDef {
    /// Parse an authoring-format (RON) game and validate it.
    pub fn from_ron(src: &str) -> Result<GameDef, DslError> {
        let g: GameDef = ron::from_str(src).map_err(|e| DslError::Parse(e.to_string()))?;
        g.check()?;
        Ok(g)
    }

    pub fn check(&self) -> Result<(), DslError> {
        let errs = validate(self);
        if errs.is_empty() {
            Ok(())
        } else {
            Err(DslError::Invalid(errs.join("\n")))
        }
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default().depth_limit(64)).expect("serialise")
    }

    /// Canonical JSON: struct field order is fixed by the type definitions, so
    /// this is stable for a given definition.
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(self).expect("serialise")
    }

    /// Hash of the full definition (including names and generator metadata).
    pub fn hash(&self) -> [u8; 32] {
        *blake3::hash(self.canonical_json().as_bytes()).as_bytes()
    }

    pub fn hash_hex(&self) -> String {
        self.hash().iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Short 64-bit id derived from the hash (used for replay headers).
    pub fn hash64(&self) -> u64 {
        u64::from_le_bytes(self.hash()[..8].try_into().unwrap())
    }
}
