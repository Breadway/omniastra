use omnia_dsl::GameDef;
use omnia_engine::*;
use omnia_search::*;
use std::sync::Arc;

fn load(name: &str) -> (Arc<Game>, GameDef) {
    let path = format!("{}/../../games/{}.ron", env!("CARGO_MANIFEST_DIR"), name);
    let def = GameDef::from_ron(&std::fs::read_to_string(path).unwrap()).unwrap();
    (Game::new(def.clone()).unwrap(), def)
}

fn winrate(name: &str, mk: &dyn Fn(&GameDef) -> Box<dyn Agent>, games: u64) -> f32 {
    let (g, def) = load(name);
    let mut score = 0.0;
    for s in 0..games {
        // alternate seats
        let (a, b): (Box<dyn Agent>, Box<dyn Agent>) = (mk(&def), Box::new(RandomAgent));
        let mut agents: Vec<Box<dyn Agent>> = if s % 2 == 0 { vec![a, b] } else { vec![b, a] };
        let (pay, _) = play_game(&g, &mut agents, 1000 + s);
        let me = (s % 2) as usize;
        score += (pay[me] + 1.0) / 2.0;
    }
    score / games as f32
}

#[test]
fn heuristic_and_mcts_beat_random() {
    for name in ["skirmish", "merchant", "gambit"] {
        let h = winrate(name, &|d| Box::new(HeuristicAgent::new(d)), 60);
        let m = winrate(name, &|d| Box::new(MctsAgent::rollout(d, 60)), 24);
        println!("{name}: heuristic vs random {h:.2}, mcts60 vs random {m:.2}");
        assert!(h > 0.5 || m > 0.5, "{name}: neither generic agent beats random");
    }
}
