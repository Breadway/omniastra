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
