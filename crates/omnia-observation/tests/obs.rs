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
        let n = st.legal_actions().len();
        st.apply(rng.below(n as u64) as usize);
    }
}

#[test]
fn stack_items_are_tokenized_with_def_actor_and_source() {
    let tk = Tokenizer::new(TokenizerConfig::default());
    let g = load("gambit");
    let mut seen_nonempty = false;
    for seed in 0..40 {
        let mut st = State::new(&g, seed);
        let mut rng = Rng::new(seed + 3);
        while !st.is_terminal() {
            let p = st.decision_maker().unwrap();
            let view = st.view(p);
            let obs = tk.observe(&mut st, p);
            assert_eq!(obs.seg.stack.len(), view.stack.len(), "one token per stack item");
            for (i, item) in view.stack.iter().enumerate() {
                seen_nonempty = true;
                let t = &obs.tokens[obs.seg.stack.start + i];
                assert_eq!(t.class, TokClass::Stack);
                assert_eq!(t.cat[0], item.def + 2, "the action def of a stack item must be visible");
                assert_eq!(t.sem[0], item.actor_rel + 1, "so must its actor");
                let rels: Vec<_> = obs.relations.iter().filter(|r| r.from as usize == obs.seg.stack.start + i).map(|r| r.kind).collect();
                assert!(rels.contains(&rel::STACK_ACTOR));
                assert_eq!(rels.contains(&rel::ACT_SOURCE), item.source.is_some());
            }
            obs.check().unwrap();
            let n = st.legal_actions().len();
            st.apply(rng.below(n as u64) as usize);
        }
    }
    assert!(seen_nonempty, "never saw a non-empty stack");
}

#[test]
fn non_object_target_ids_cannot_collide_with_template_ids() {
    // template ids are 1..=MAX_TEMPLATES; every special id must lie above them and fit the model vocabulary
    let templates_top = omnia_dsl::MAX_TEMPLATES as u16;
    for id in [CAT_TARGET_PLAYER, CAT_TARGET_PLAYER + omnia_dsl::MAX_PLAYERS as u16 - 1, CAT_TARGET_NUM, CAT_TARGET_HIDDEN] {
        assert!(id > templates_top, "id {id} overlaps template ids");
    }
    assert_eq!(CAT_MAX, CAT_TARGET_HIDDEN);
    assert!(CAT_TARGET_PLAYER + (omnia_dsl::MAX_PLAYERS as u16) <= CAT_TARGET_NUM, "player target ids overlap the number id");
}
