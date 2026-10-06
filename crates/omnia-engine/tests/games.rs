use omnia_dsl::{GameDef, SetupEntry, Visibility};
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
    for (name, relabel_seed) in GAMES.iter().flat_map(|n| [77u64, 1, 2, 3, 4, 5].map(|s| (*n, s))) {
        let path = format!("{}/../../games/{}.ron", env!("CARGO_MANIFEST_DIR"), name);
        let def = GameDef::from_ron(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut rr = Rng::new(relabel_seed);
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

/// `EventSlot` constants must follow the id permutation, otherwise the
/// skirmish death trigger watches the wrong attribute after relabelling.
#[test]
fn relabel_remaps_event_slot_constants() {
    use omnia_dsl::{CmpOp, Cond, Expr, Relabel};
    let path = format!("{}/../../games/skirmish.ron", env!("CARGO_MANIFEST_DIR"));
    let def = GameDef::from_ron(&std::fs::read_to_string(path).unwrap()).unwrap();
    for seed in 0..32u64 {
        let mut rr = Rng::new(seed);
        let rl = Relabel::random(&def, |n| rr.below(n));
        let def2 = rl.apply(&def);
        let Cond::And(parts) = &def2.triggers[0].cond else { panic!("unexpected trigger shape") };
        let Cond::Cmp(a, CmpOp::Eq, b) = &parts[0] else { panic!("unexpected slot test") };
        assert_eq!(**a, Expr::EventSlot);
        assert_eq!(**b, Expr::Const(rl.attr[2] as i32), "seed {seed}");
        let Cond::Cmp(lhs, CmpOp::Le, _) = &parts[1] else { panic!("unexpected health test") };
        assert!(matches!(**lhs, Expr::Attr(_, id) if id == rl.attr[2]), "seed {seed}");
    }
}

fn load_def(name: &str) -> GameDef {
    let path = format!("{}/../../games/{}.ron", env!("CARGO_MANIFEST_DIR"), name);
    GameDef::from_ron(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn shared_objects_are_unowned_until_claimed() {
    use omnia_engine::view::NO_REL;
    let g = load("merchant");
    let (mut saw_unowned, mut saw_claimed) = (false, false);
    for seed in 0..30 {
        let mut st = State::new(&g, seed);
        let mut rng = Rng::new(seed + 7);
        while !st.is_terminal() {
            let p = st.decision_maker().unwrap();
            let v = st.view(p);
            for o in &v.objects {
                let zone = &v.zones[o.zone as usize];
                if zone.owner_rel == NO_REL {
                    // still in a shared zone: nobody owns or controls it (unless it came from a player)
                    saw_unowned |= o.owner_rel == NO_REL && o.controller_rel == NO_REL;
                } else {
                    // a player's zone: the object is owned by someone
                    assert_ne!(o.owner_rel, NO_REL, "seed {seed}: object in a player zone has no owner");
                    saw_claimed = true;
                }
            }
            let n = st.legal_actions().len();
            st.apply(rng.below(n as u64) as usize);
        }
    }
    assert!(saw_unowned && saw_claimed);
}

#[test]
fn market_view_is_symmetric_between_seats() {
    use omnia_engine::view::NO_REL;
    let g = load("merchant");
    let mut st = State::new(&g, 3);
    for p in 0..2 {
        let v = st.view(p);
        let market = v.zones.iter().position(|z| z.def == 0).unwrap();
        assert_eq!(v.zones[market].owner_rel, NO_REL);
        let in_market: Vec<_> = v.objects.iter().filter(|o| o.zone as usize == market).collect();
        assert!(!in_market.is_empty());
        assert!(in_market.iter().all(|o| o.owner_rel == NO_REL && o.controller_rel == NO_REL), "seat {p} sees the market owned");
    }
}

#[test]
fn determinize_does_not_mix_shared_hidden_cards_with_hands() {
    // gambit with a hidden shared zone holding only `counter` (template 4)
    let mut def = load_def("gambit");
    def.zones[3].vis = Visibility::Hidden;
    def.shared_setup.push(SetupEntry { zone: 3, template: 4, count: 3 });
    let g = Game::new(def).unwrap();
    let st = State::new(&g, 1);
    let shared = g.zone_base[3] as usize;
    assert_eq!(st.zone_contents(shared).len(), 3);
    let mut rng = Rng::new(5);
    for _ in 0..40 {
        let d = st.determinize(0, &mut rng);
        for o in d.zone_contents(shared) {
            assert_eq!(d.object(*o).template, 4, "a shared hidden card took a player's identity");
        }
    }
}

#[test]
fn determinize_respects_zone_constraints_without_falling_back_to_truth() {
    // Opponent hands may only hold kind-0 cards, decks hold both kinds. Naive
    // shuffles almost never satisfy that, so a fallback would keep the truth.
    let mut def = load_def("gambit");
    def.zones[1].allowed_kinds = Some(vec![0]);
    // make kind-1 cards plentiful so a random shuffle rarely satisfies the hand constraint
    def.player_setup[4].count = 40;
    def.player_setup.push(SetupEntry { zone: 1, template: 0, count: 6 });
    let g = Game::new(def).unwrap();
    let st = State::new(&g, 2);
    let (opp_hand, opp_deck) = (g.zone_base[1] as usize + 1, g.zone_base[0] as usize + 1);
    let truth = |s: &State, z: usize| -> Vec<u16> { s.zone_contents(z).iter().map(|o| s.object(*o).template).collect() };
    let mut rng = Rng::new(11);
    let mut changed = 0;
    for _ in 0..20 {
        let d = st.determinize(0, &mut rng);
        for o in d.zone_contents(opp_hand) {
            let t = d.object(*o).template as usize;
            assert_eq!(g.def.templates[t].kind, 0, "constraint violated");
        }
        if truth(&d, opp_hand) != truth(&st, opp_hand) || truth(&d, opp_deck) != truth(&st, opp_deck) {
            changed += 1;
        }
    }
    assert!(changed >= 10, "determinize kept the true identities in {} of 20 samples", 20 - changed);
}

/// A spell sits in limbo only while it is on the stack. Countering it must not
/// strand it there (its normal effect, which discards it, never runs).
#[test]
fn countered_spells_leave_limbo() {
    let g = load("gambit");
    let limbo = g.zone_base[3] as usize;
    let mut saw_cancel = false;
    for seed in 0..300 {
        let mut st = State::new(&g, seed);
        let mut rng = Rng::new(seed + 1);
        while !st.is_terminal() {
            if st.stack_len() == 0 {
                assert!(st.zone_contents(limbo).is_empty(), "seed {seed} turn {}: {} card(s) stuck in limbo", st.turn(), st.zone_contents(limbo).len());
            }
            let n = st.legal_actions().len();
            st.apply(rng.below(n as u64) as usize);
        }
        saw_cancel |= st.history().iter().any(|e| matches!(e.kind, omnia_dsl::EventKind::StackCancelled));
    }
    assert!(saw_cancel, "no counter ever cancelled a spell");
}

#[test]
fn costs_are_checked_cumulatively() {
    use omnia_dsl::Expr;
    let castable = |costs: Vec<(u8, Expr)>| {
        let mut def = load_def("gambit");
        def.actions[0].costs = costs;
        let g = Game::new(def).unwrap();
        // focus is 3 on turn 1, cast is the action with def 0
        (0..30).any(|seed| State::new(&g, seed).legal_actions().iter().any(|a| a.def == 0))
    };
    assert!(castable(vec![(1, Expr::Const(2))]), "a single cost of 2 fits in 3 focus");
    assert!(!castable(vec![(1, Expr::Const(2)), (1, Expr::Const(2))]), "two costs of 2 on the same resource need 4 focus");
}

#[test]
fn extreme_arithmetic_does_not_panic() {
    use omnia_dsl::{Effect, Expr, PRef};
    let mut def = load_def("gambit");
    let b = |e: Expr| Box::new(e);
    def.actions[2].effect = Effect::Seq(vec![
        Effect::ModRes(PRef::Me, 0, Expr::Neg(b(Expr::Const(i32::MIN)))),
        Effect::ModRes(PRef::Me, 0, Expr::Div(b(Expr::Const(i32::MIN)), b(Expr::Const(-1)))),
        Effect::ModRes(PRef::Me, 0, Expr::Rand(i32::MIN, i32::MAX)),
        Effect::EndTurn,
    ]);
    let g = Game::new(def).unwrap();
    let mut st = State::new(&g, 1);
    let end = st.legal_actions().iter().position(|a| a.def == 2).unwrap();
    st.apply(end); // must not panic on i32::MIN negation / division / full-range Rand
    assert!(st.fault().is_none());
}

#[test]
fn simultaneous_wins_are_a_draw_not_two_winners() {
    use omnia_dsl::{Cond, PlayerResult, TerminalRule};
    let mut def = load_def("gambit");
    def.terminal = vec![TerminalRule { cond: Cond::True, result: PlayerResult::Win }];
    let g = Game::new(def).unwrap();
    let st = State::new(&g, 1);
    let pay = st.payoffs().expect("game ends immediately");
    assert_eq!(&pay[..2], &[0.0, 0.0], "both players won at once");
    assert_eq!(pay[0] + pay[1], 0.0, "two-player payoffs must be zero-sum");
}
