use omnia_gen::*;

#[test]
fn families_generate_valid_games_with_measured_stats() {
    let cfg = ValidationConfig::default();
    for fam in Family::ALL {
        for variant in [Variant::Normal, Variant::AltVictory, Variant::Inverted] {
            if !fam.supports(variant) {
                continue;
            }
            let (mut ok, mut tot) = (0, 0);
            let mut sample = None;
            for seed in 0..12u64 {
                let spec = GenSpec { family: fam, seed, variant, relabel: true, random_reward: false };
                let def = generate(&spec);
                def.check().unwrap_or_else(|e| panic!("{:?} {:?} seed {seed}: static validation failed: {e}", fam, variant));
                tot += 1;
                match measure(&def, &cfg) {
                    Ok(s) => {
                        ok += 1;
                        if sample.is_none() {
                            sample = Some(s);
                        }
                    }
                    Err(e) => println!("  reject {:?} {:?} {seed}: {e}", fam, variant),
                }
            }
            let s = sample.map(|s| format!("dec {:.0} br {:.1} p0 {:.2} p1 {:.2} to {:.2} h {:.2} m {:.2}", s.avg_decisions, s.avg_branching, s.p0_win_rate, s.p1_win_rate, s.timeout_rate, s.heuristic_vs_random.unwrap_or(-1.0), s.mcts_vs_random.unwrap_or(-1.0))).unwrap_or_default();
            println!("{:?}/{:?}: accepted {ok}/{tot}  {s}", fam, variant);
        }
    }
}

#[test]
fn countered_spells_do_not_strand_cards_in_generated_stack_games() {
    use omnia_engine::*;
    let mut cancels = 0;
    for seed in 0..12u64 {
        let spec = GenSpec { family: Family::Stack, seed, variant: Variant::Normal, relabel: true, random_reward: false };
        let def = generate(&spec);
        let limbo_def = def.zones.iter().position(|z| z.name == "limbo").unwrap();
        let g = Game::new(def).unwrap();
        let limbo = g.zone_base[limbo_def] as usize;
        for play in 0..40u64 {
            let mut st = State::new(&g, play);
            let mut rng = Rng::new(play + 100 * seed);
            while !st.is_terminal() {
                if st.stack_len() == 0 {
                    assert!(st.zone_contents(limbo).is_empty(), "stack seed {seed} play {play}: card stuck in limbo");
                }
                let n = st.legal_actions().len();
                st.apply(rng.below(n as u64) as usize);
            }
            cancels += st.history().iter().filter(|e| matches!(e.kind, omnia_dsl::EventKind::StackCancelled)).count();
        }
    }
    assert!(cancels > 0, "no generated stack game ever cancelled a spell");
}
