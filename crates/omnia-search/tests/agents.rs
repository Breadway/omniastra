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
        // each agent is checked on its own (seeds are fixed, so this is deterministic)
        assert!(m > 0.65, "{name}: mcts60 should clearly beat random, got {m:.2}");
        // the 1-ply heuristic is weak in skirmish (about even with random), solid elsewhere
        let floor = if name == "skirmish" { 0.4 } else { 0.6 };
        assert!(h > floor, "{name}: heuristic should score above {floor} vs random, got {h:.2}");
    }
}

/// Counts evaluator calls so tests can bound the search cost.
struct CountingEval {
    inner: RolloutEval,
    calls: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl Evaluator for CountingEval {
    fn evaluate(&mut self, st: &mut State, rng: &mut Rng) -> Eval {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.evaluate(st, rng)
    }
}

#[test]
fn mcts_uses_at_most_one_evaluation_per_simulation() {
    use std::sync::atomic::{AtomicU32, Ordering};
    let (g, def) = load("skirmish");
    for sims in [8u32, 30, 60] {
        let calls = Arc::new(AtomicU32::new(0));
        let mut mcts = Mcts::new(MctsConfig { sims, ..Default::default() }, CountingEval { inner: RolloutEval::new(&def), calls: calls.clone() });
        let mut st = State::new(&g, 4);
        let mut rng = Rng::new(9);
        let r = mcts.search(&mut st, &mut rng);
        assert_eq!(r.policy.len(), st.legal_actions().len());
        assert!(calls.load(Ordering::SeqCst) <= sims, "{sims} sims used {} evaluations", calls.load(Ordering::SeqCst));
    }
}

/// A game where one action peeks at the hidden top card: "gamble" pays +4 if the
/// top card is a burst (cost attr 3) and 0 otherwise, "safe" always pays +2.
/// A heuristic agent that reads the true state would gamble exactly when the
/// real top card is a burst.
fn gamble_game() -> Arc<Game> {
    use omnia_dsl::*;
    let (_, mut def) = load("gambit");
    def.player_setup = vec![SetupEntry { zone: 0, template: 0, count: 1 }, SetupEntry { zone: 0, template: 1, count: 1 }];
    def.setup = Effect::ForEachPlayer(PSel::All, Box::new(Effect::Shuffle(ZRef { who: ZWho::Me, zone: 0 })));
    def.phases[0].on_enter = Effect::NoOp;
    let top = Sel::Zones { zones: vec![ZRef { who: ZWho::Me, zone: 0 }], filter: Cond::True, order: SelOrder::Top(Box::new(Expr::Const(1))) };
    let payoff = Expr::Mul(Box::new(Expr::Sub(Box::new(Expr::Attr(ORef::Iter, 0)), Box::new(Expr::Const(2)))), Box::new(Expr::Const(4)));
    let gamble = ActionDef { name: "gamble".into(), class: 0, phases: vec![1], timing: Timing::Main, source: None, targets: vec![], costs: vec![], require: Cond::True, on_use: Effect::NoOp, on_cancel: Effect::NoOp, effect: Effect::ForEach(top, Box::new(Effect::ModRes(PRef::Me, 0, payoff))), stack: false };
    let safe = ActionDef { name: "safe".into(), effect: Effect::ModRes(PRef::Me, 0, Expr::Const(2)), ..gamble.clone() };
    let safe = ActionDef { class: 1, ..safe };
    let end_turn = def.actions[2].clone();
    def.actions = vec![gamble, safe, end_turn];
    def.check().unwrap();
    Game::new(def).unwrap()
}

#[test]
fn heuristic_agent_does_not_peek_at_hidden_draws() {
    let g = gamble_game();
    let deck0 = g.zone_base[0] as usize;
    // one seed whose top card is the burst (template 1), one where it is not
    let top_is_burst = |seed: u64| {
        let st = State::new(&g, seed);
        st.object(st.zone_contents(deck0)[0]).template == 1
    };
    let good = (0..64).find(|s| top_is_burst(*s)).unwrap();
    let bad = (0..64).find(|s| !top_is_burst(*s)).unwrap();
    let gamble_rate = |seed: u64| {
        let mut gambles = 0;
        for k in 0..200u64 {
            let mut st = State::new(&g, seed);
            let mut agent = HeuristicAgent::new(&g.def);
            agent.epsilon = 0.0;
            let mut rng = Rng::new(1000 + k);
            let i = agent.act(&mut st, &mut rng);
            if st.legal_actions()[i].def == 0 {
                gambles += 1;
            }
        }
        gambles as f32 / 200.0
    };
    let (a, b) = (gamble_rate(good), gamble_rate(bad));
    // a cheating agent gambles 100% of the time in one world and 0% in the other
    assert!((a - b).abs() < 0.25, "choice depends on the hidden top card: {a} vs {b}");
}
