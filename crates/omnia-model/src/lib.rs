//! OmniAstra: variable-length relational Transformer policy/value model.
//!
//! Generic over the Burn backend. Nothing here refers to individual games.

pub mod config;
pub mod embed;
pub mod layers;
pub mod model;
pub mod tensors;

pub use config::*;
pub use model::*;
pub use tensors::*;
