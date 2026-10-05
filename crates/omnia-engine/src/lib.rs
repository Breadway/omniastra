//! OmniEngine: deterministic, headless interpreter for OmniDSL games.
//!
//! Nothing in this crate is specific to any game; games are data
//! ([`omnia_dsl::GameDef`]) compiled into a [`Game`] and executed as [`State`].

mod exec;
mod flow;
pub mod replay;
pub mod rng;
mod state;
mod types;
pub mod view;

pub use replay::Replay;
pub use rng::Rng;
pub use state::State;
pub use types::*;
pub use view::*;
