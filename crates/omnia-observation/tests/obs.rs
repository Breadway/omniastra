use omnia_dsl::GameDef;
use omnia_engine::*;
use omnia_observation::*;
use std::sync::Arc;

fn load(name: &str) -> Arc<Game> {
    let path = format!("{}/../../games/{}.ron", env!("CARGO_MANIFEST_DIR"), name);
    Game::new(GameDef::from_ron(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap()
}
const GAMES: [&str; 3] = ["skirmish", "merchant", "gambit"];

#[test]
fn observations_are_well_formed_and_variable_length() {
    let tk = Tokenizer::new(TokenizerConfig::default());
    for name in GAMES {
        let g = load(name);
        let mut st = State::new(&g, 5);
        let mut rng = Rng::new(2);
        let mut lens = std::collections::BTreeSet::new();
        let mut alens = std::collections::BTreeSet::new();
        while !st.is_terminal() {
            let p = st.decision_maker().unwrap();
            let obs = tk.observe(&mut st, p);
            obs.check().unwrap();
            assert_eq!(obs.n_actions(), st.legal_actions().len(), "action tokens must align with legal actions");
            lens.insert(obs.n_state());
            alens.insert(obs.n_actions());
            let n = st.legal_actions().len();
            st.apply(rng.below(n as u64) as usize);
        }
        assert!(lens.len() > 3, "{name}: state size should vary");
        println!("{name}: state sizes {}..{} action counts {:?}", lens.iter().next().unwrap(), lens.iter().last().unwrap(), alens);
    }
}

#[test]
fn unordered_zone_order_does_not_change_observation() {
    let tk = Tokenizer::new(TokenizerConfig::default());
    let g = load("skirmish");
    let mut st = State::new(&g, 9);
    let mut rng = Rng::new(3);
    for _ in 0..12 {
        if st.is_terminal() {
            break;
        }
        let p = st.decision_maker().unwrap();
        let a = serde_json::to_string(&tk.observe(&mut st, p)).unwrap();
        let mut st2 = st.clone();
        // hand zone instances are the unordered ones (zone def 1)
        for inst in 0..g.n_instances {
            if g.inst[inst].0 == 1 {
                st2.permute_zone_for_test(inst, &mut rng);
            }
        }
        // Permuting the true internal order must not change what the player sees,
        // except for the (arbitrary) action order which follows source order.
        let o1 = tk.observe(&mut st, p);
        let o2 = tk.observe(&mut st2, p);
        assert_eq!(o1.state_tokens().len(), o2.state_tokens().len());
        let s1 = serde_json::to_string(o1.state_tokens()).unwrap();
        let s2 = serde_json::to_string(o2.state_tokens()).unwrap();
        assert_eq!(s1, s2);
        let _ = a;
        let n = st.legal_actions().len();
        st.apply(rng.below(n as u64) as usize);
    }
}
