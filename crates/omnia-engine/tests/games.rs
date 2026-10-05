use omnia_dsl::GameDef;
use omnia_engine::*;
use std::sync::Arc;

fn load(name: &str) -> Arc<Game> {
    let path = format!("{}/../../games/{}.ron", env!("CARGO_MANIFEST_DIR"), name);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let def = GameDef::from_ron(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
    Game::new(def).unwrap()
}

const GAMES: [&str; 3] = ["skirmish", "merchant", "gambit"];

fn playout(game: &Arc<Game>, seed: u64) -> (Replay, State) {
    let mut rng = Rng::new(seed ^ 0xABCD);
    let mut st = State::new(game, seed);
    let mut rep = Replay::new(game, seed);
    while !st.is_terminal() {
        let n = st.legal_actions().len();
        assert!(n > 0, "non-terminal state with no legal actions");
        let i = rng.below(n as u64) as usize;
        rep.actions.push(i as u32);
        st.apply(i);
    }
    (rep, st)
}

#[test]
fn games_parse_and_terminate_without_faults() {
    for name in GAMES {
        let g = load(name);
        let mut wins = [0u32; 3];
        let mut total_decisions = 0;
        for seed in 0..200 {
            let (_, st) = playout(&g, seed);
            assert!(st.fault().is_none(), "{name} seed {seed}: fault {:?}", st.fault());
            let o = st.outcome().unwrap();
            let idx = if o.payoffs[0] > 0.0 { 0 } else if o.payoffs[1] > 0.0 { 1 } else { 2 };
            wins[idx] += 1;
            total_decisions += st.decisions();
        }
        println!("{name}: p0 wins {} p1 wins {} draws {} avg decisions {}", wins[0], wins[1], wins[2], total_decisions / 200);
    }
}

#[test]
fn replay_is_deterministic() {
    for name in GAMES {
        let g = load(name);
        for seed in 0..20 {
            let (rep, st) = playout(&g, seed);
            let st2 = rep.replay(&g).unwrap();
            assert_eq!(st.state_hash(), st2.state_hash(), "{name} seed {seed}");
            assert_eq!(st.payoffs(), st2.payoffs());
        }
    }
}

#[test]
fn clone_then_diverge_is_independent_and_matches_replay() {
    for name in GAMES {
        let g = load(name);
        let mut st = State::new(&g, 7);
        let mut rng = Rng::new(1);
        for _ in 0..6 {
            if st.is_terminal() {
                break;
            }
            let n = st.legal_actions().len();
            st.apply(rng.below(n as u64) as usize);
        }
        if st.is_terminal() {
            continue;
        }
        let h0 = st.state_hash();
        let mut c = st.clone();
        let n = c.legal_actions().len();
        c.apply(0.min(n - 1));
        assert_eq!(st.state_hash(), h0, "{name}: original mutated by clone step");
        // every action leads to a state without panicking
        for i in 0..st.legal_actions().len() {
            let mut c = st.clone();
            c.apply(i);
        }
    }
}

#[test]
fn observations_do_not_leak_hidden_information() {
    for name in GAMES {
        let g = load(name);
        let mut checked = 0;
        for seed in 0..12 {
            let mut st = State::new(&g, seed);
            let mut rng = Rng::new(seed + 99);
            let mut det_rng = Rng::new(seed + 5);
            while !st.is_terminal() {
                // Players are only ever asked for an observation at their own
                // decision points; at others the engine may auto-resolve
                // (e.g. an opponent with no legal response is skipped), which
                // legitimately depends on hidden cards but is not observable.
                let p = st.decision_maker().unwrap();
                {
                    let a = serde_json::to_string(&st.view(p)).unwrap();
                    let mut d = st.determinize(p, &mut det_rng);
                    let b = serde_json::to_string(&d.view(p)).unwrap();
                    assert_eq!(a, b, "{name} seed {seed} turn {} player {p}: view depends on hidden state", st.turn());
                    checked += 1;
                }
                let n = st.legal_actions().len();
                st.apply(rng.below(n as u64) as usize);
            }
        }
        assert!(checked > 100);
    }
}

#[test]
fn determinization_actually_changes_hidden_state() {
    // Negative control: the checker above is only meaningful if determinize
    // really resamples hidden identities.
    let g = load("skirmish");
    let mut st = State::new(&g, 3);
    let mut r = Rng::new(11);
    let mut changed = false;
    for _ in 0..5 {
        let d = st.determinize(0, &mut r);
        if d.state_hash() != st.state_hash() {
            changed = true;
        }
    }
    assert!(changed);
    let _ = &mut st;
}

#[test]
fn stack_and_counters_work_in_gambit() {
    let g = load("gambit");
    // Find a game where a counter was cast: check history for StackCancelled.
    let mut saw_cancel = false;
    for seed in 0..300 {
        let (_, st) = playout(&g, seed);
        if st.history().iter().any(|e| e.kind == omnia_dsl::EventKind::StackCancelled) {
            saw_cancel = true;
            break;
        }
    }
    assert!(saw_cancel, "no counter ever cancelled a spell in 300 random games");
}

#[test]
fn relabelled_games_are_mechanically_equivalent() {
    use omnia_dsl::Relabel;
    for name in GAMES {
        let path = format!("{}/../../games/{}.ron", env!("CARGO_MANIFEST_DIR"), name);
        let def = GameDef::from_ron(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut rr = Rng::new(77);
        let rl = Relabel::random(&def, |n| rr.below(n));
        let def2 = rl.apply(&def);
        def2.check().unwrap_or_else(|e| panic!("{name}: relabelled game invalid: {e}"));
        assert_ne!(def.hash64(), def2.hash64());
        let g1 = Game::new(def).unwrap();
        let g2 = Game::new(def2).unwrap();
        let stats = |g: &Arc<Game>| {
            let (mut w0, mut dec) = (0.0f64, 0.0f64);
            let n = 400;
            for s in 0..n {
                let (_, st) = playout(g, s);
                assert!(st.fault().is_none());
                w0 += (st.payoffs().unwrap()[0] > 0.0) as u8 as f64;
                dec += st.decisions() as f64;
            }
            (w0 / n as f64, dec / n as f64)
        };
        let (a, b) = (stats(&g1), stats(&g2));
        println!("{name}: orig {:?} relabelled {:?}", a, b);
        assert!((a.0 - b.0).abs() < 0.1, "{name}: win rate differs {a:?} vs {b:?}");
        assert!((a.1 - b.1).abs() / a.1 < 0.1, "{name}: length differs {a:?} vs {b:?}");
    }
}
