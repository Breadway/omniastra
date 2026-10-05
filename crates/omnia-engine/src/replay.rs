use crate::{Game, State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A game is exactly reproducible from (definition hash, seed, action indices).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Replay {
    pub game_hash: u64,
    pub game_name: String,
    pub seed: u64,
    /// Index into the (deterministically ordered) legal-action list at each decision.
    pub actions: Vec<u32>,
}

impl Replay {
    pub fn new(game: &Game, seed: u64) -> Replay {
        Replay { game_hash: game.hash64, game_name: game.def.name.clone(), seed, actions: vec![] }
    }

    pub fn replay(&self, game: &Arc<Game>) -> Result<State, String> {
        if game.hash64 != self.game_hash {
            return Err(format!("game hash mismatch: replay {:016x} vs game {:016x}", self.game_hash, game.hash64));
        }
        let mut st = State::new(game, self.seed);
        for (i, a) in self.actions.iter().enumerate() {
            if st.is_terminal() || (*a as usize) >= st.legal_actions().len() {
                return Err(format!("invalid action {} at step {}", a, i));
            }
            st.apply(*a as usize);
        }
        Ok(st)
    }
}
