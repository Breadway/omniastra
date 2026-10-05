//! `omnia`: command-line front end for OmniAstra.

mod backend;
mod describe;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use omnia_dsl::GameDef;
use omnia_engine::*;
use omnia_gen::*;
use omnia_observation::{Tokenizer, TokenizerConfig};
use omnia_search::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "omnia", about = "OmniAstra: cross-game strategic learning platform")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Game definition tools.
    #[command(subcommand)]
    Game(GameCmd),
    /// Play games between agents and print results (optionally a trace).
    Play {
        game: PathBuf,
        #[arg(long, default_value = "random")]
        p1: String,
        #[arg(long, default_value = "random")]
        p2: String,
        #[arg(long, default_value_t = 1)]
        games: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Print every decision of the first game.
        #[arg(long)]
        trace: bool,
        /// Write the first game's replay to this file.
        #[arg(long)]
        save_replay: Option<PathBuf>,
    },
    /// Random-play statistics for a game.
    Simulate {
        game: PathBuf,
        #[arg(long, default_value_t = 1000)]
        games: u32,
    },
    /// Step through a replay file.
    Replay { file: PathBuf, #[arg(long)] game: PathBuf },
    /// Throughput benchmarks (engine, observation, model).
    Benchmark {
        #[arg(long)]
        game: Option<PathBuf>,
        #[arg(long, default_value = "tiny")]
        model: String,
        #[arg(long, default_value_t = 32)]
        batch: usize,
    },
    /// Model utilities.
    #[command(subcommand)]
    Model(ModelCmd),
    /// Run the central transfer experiment from a JSON config.
    Transfer {
        config: PathBuf,
        #[arg(long, default_value = "games")]
        games_dir: PathBuf,
        /// Number of devices (data-parallel shards) to train on.
        #[arg(long, default_value_t = 1)]
        devices: usize,
    },
    /// Multi-game training from a JSON job config; saves a checkpoint.
    Train {
        config: PathBuf,
        #[arg(long, default_value = "games")]
        games_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        devices: usize,
    },
    /// Search-enhanced self-play with a checkpoint league, from a JSON job config.
    Selfplay {
        config: PathBuf,
        #[arg(long, default_value = "games")]
        games_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        devices: usize,
    },
    /// Evaluate a checkpoint's raw policy vs a panel on a game file.
    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value = "nano")]
        model: String,
        #[arg(long)]
        game: PathBuf,
        #[arg(long, default_value = "random,heuristic,mcts:16")]
        opponents: String,
        #[arg(long, default_value_t = 50)]
        games: u32,
        #[arg(long, default_value_t = 0)]
        game_idx: u32,
    },
    /// Check single- vs multi-shard gradient parity (correctness test).
    ParityCheck {
        #[arg(long, default_value_t = 3)]
        shards: usize,
    },
}

#[derive(Subcommand)]
enum GameCmd {
    /// Validate a game file (static checks; optionally random-play statistics).
    Validate { path: PathBuf, #[arg(long)] stats: bool },
    /// Summarise a game: structure, descriptors, hash.
    Inspect { path: PathBuf },
    /// Generate a validated set of games of one family.
    Generate {
        #[arg(long)]
        family: String,
        #[arg(long, default_value = "normal")]
        variant: String,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 5)]
        count: usize,
        #[arg(long)]
        out: PathBuf,
        /// Keep canonical ids instead of randomly relabelling them.
        #[arg(long)]
        canonical_ids: bool,
        #[arg(long)]
        random_reward: bool,
    },
    /// Derive a control from an existing game file.
    Control {
        path: PathBuf,
        /// reskin | random-reward
        kind: String,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum ModelCmd {
    /// Print architecture size presets and parameter counts.
    Inspect {
        #[arg(long, default_value = "tiny")]
        size: String,
    },
}

fn load_def(p: &Path) -> Result<GameDef> {
    Ok(GameDef::from_ron(&std::fs::read_to_string(p).map_err(|e| anyhow!("{}: {e}", p.display()))?)?)
}

fn load(p: &Path) -> Result<Arc<Game>> {
    Ok(Game::new(load_def(p)?)?)
}

fn agent_from(spec: &str, game: &Arc<Game>) -> Result<Box<dyn Agent>> {
    Ok(if spec == "random" {
        Box::new(RandomAgent)
    } else if spec == "heuristic" {
        Box::new(HeuristicAgent::new(&game.def))
    } else if let Some(n) = spec.strip_prefix("mcts:") {
        Box::new(MctsAgent::rollout(&game.def, n.parse()?))
    } else {
        return Err(anyhow!("unknown agent '{spec}' (random | heuristic | mcts:<sims>)"));
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Game(g) => game_cmd(g),
        Cmd::Play { game, p1, p2, games, seed, trace, save_replay } => {
            let g = load(&game)?;
            let (mut w, mut l, mut d) = (0, 0, 0);
            let mut dec = 0.0;
            for i in 0..games {
                let mut agents = vec![agent_from(&p1, &g)?, agent_from(&p2, &g)?];
                let s = seed + i as u64;
                let mut st = State::new(&g, s);
                let mut rng = Rng::new(s ^ 0x5EED);
                let mut rep = Replay::new(&g, s);
                while !st.is_terminal() {
                    let p = st.decision_maker().unwrap() as usize;
                    let idx = agents[p].act(&mut st, &mut rng);
                    if trace && i == 0 {
                        println!("T{} P{} [{}] {}", st.turn(), p, agents[p].name(), describe::action(&st, &st.legal_actions()[idx]));
                    }
                    rep.actions.push(idx as u32);
                    st.apply(idx);
                }
                let pay = st.payoffs().unwrap();
                if trace && i == 0 {
                    println!("result: {:?} ({:?})", &pay[..g.n() as usize], st.outcome().unwrap().reason);
                    describe::state(&st);
                }
                if let (0, Some(path)) = (i, &save_replay) {
                    std::fs::write(path, serde_json::to_string_pretty(&rep)?)?;
                }
                dec += st.decisions() as f64;
                if pay[0] > 0.0 {
                    w += 1
                } else if pay[1] > 0.0 {
                    l += 1
                } else {
                    d += 1
                }
            }
            println!("{p1} (P1) vs {p2} (P2): {w} wins, {l} losses, {d} draws over {games} games; avg decisions {:.1}", dec / games as f64);
            Ok(())
        }
        Cmd::Simulate { game, games } => {
            let g = load(&game)?;
            let t = Instant::now();
            let (mut w0, mut w1, mut dr, mut dec, mut to) = (0, 0, 0, 0u64, 0);
            for s in 0..games as u64 {
                let mut st = State::new(&g, s);
                let mut rng = Rng::new(s + 17);
                while !st.is_terminal() {
                    let n = st.legal_actions().len();
                    st.apply(rng.below(n as u64) as usize);
                }
                let o = st.outcome().unwrap();
                dec += st.decisions() as u64;
                if matches!(o.reason, EndReason::Timeout) {
                    to += 1;
                }
                if o.payoffs[0] > 0.0 {
                    w0 += 1
                } else if o.payoffs[1] > 0.0 {
                    w1 += 1
                } else {
                    dr += 1
                }
            }
            let el = t.elapsed().as_secs_f64();
            println!("{} games in {:.2}s ({:.0} games/s, {:.0} decisions/s): p0 {w0} p1 {w1} draw {dr} timeouts {to}; avg decisions {:.1}", games, el, games as f64 / el, dec as f64 / el, dec as f64 / games as f64);
            Ok(())
        }
        Cmd::Replay { file, game } => {
            let g = load(&game)?;
            let rep: Replay = serde_json::from_str(&std::fs::read_to_string(file)?)?;
            let mut st = State::new(&g, rep.seed);
            if g.hash64 != rep.game_hash {
                return Err(anyhow!("game hash mismatch (replay {:016x}, game {:016x})", rep.game_hash, g.hash64));
            }
            for (i, a) in rep.actions.iter().enumerate() {
                let act = st.legal_actions()[*a as usize].clone();
                println!("{:>3}. T{} P{}: {}", i, st.turn(), act.actor, describe::action(&st, &act));
                st.apply(*a as usize);
            }
            println!("final: {:?}", st.payoffs().map(|p| p[..g.n() as usize].to_vec()));
            describe::state(&st);
            Ok(())
        }
        Cmd::Benchmark { game, model, batch } => benchmark(game, &model, batch),
        Cmd::Model(ModelCmd::Inspect { size }) => backend::model_inspect(&size),
        Cmd::Transfer { config, games_dir, devices } => backend::run_transfer(&config, &games_dir, devices),
        Cmd::ParityCheck { shards } => backend::parity_check(shards),
        Cmd::Train { config, games_dir, devices } => backend::run_train(&config, &games_dir, devices),
        Cmd::Selfplay { config, games_dir, devices } => backend::run_selfplay(&config, &games_dir, devices),
        Cmd::Evaluate { checkpoint, model, game, opponents, games, game_idx } => {
            let g = load(&game)?;
            let opps: Vec<String> = opponents.split(',').map(|s| s.to_string()).collect();
            backend::run_eval(&checkpoint, &model, g, game_idx, &opps, games)
        }
    }
}

fn game_cmd(g: GameCmd) -> Result<()> {
    match g {
        GameCmd::Validate { path, stats } => {
            let def = load_def(&path)?;
            println!("{}: static validation OK (hash {})", def.name, def.hash_hex());
            if stats {
                let s = measure(&def, &ValidationConfig::default()).map_err(|e| anyhow!(e))?;
                println!("{}", serde_json::to_string_pretty(&s)?);
            }
            Ok(())
        }
        GameCmd::Inspect { path } => describe::game(&load_def(&path)?),
        GameCmd::Generate { family, variant, seed, count, out, canonical_ids, random_reward } => {
            let fam = Family::parse(&family).ok_or_else(|| anyhow!("unknown family {family}"))?;
            let v = match variant.as_str() {
                "alt_victory" => Variant::AltVictory,
                "inverted" => Variant::Inverted,
                _ => Variant::Normal,
            };
            if !fam.supports(v) {
                return Err(anyhow!("family {family} has no variant {variant}"));
            }
            let entries = generate_set(&out, fam, v, !canonical_ids, random_reward, seed, count, &ValidationConfig::default());
            std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&GameSet { name: format!("{family}-{variant}"), games: entries.clone() })?)?;
            for e in &entries {
                println!(
                    "{:<32} tries {:<2} dec {:>5.1} branch {:>4.1} p0 {:.2} mcts-vs-random {:?}",
                    e.id, e.tries, e.stats.avg_decisions, e.stats.avg_branching, e.stats.p0_win_rate, e.stats.mcts_vs_random
                );
            }
            println!("{} games written to {}", entries.len(), out.display());
            Ok(())
        }
        GameCmd::Control { path, kind, seed, out } => {
            let def = load_def(&path)?;
            let c = match kind.as_str() {
                "reskin" => reskin(&def, seed),
                "random-reward" => random_reward(&def),
                _ => return Err(anyhow!("control kind must be reskin | random-reward")),
            };
            std::fs::write(&out, c.to_ron())?;
            println!("wrote {} ({})", out.display(), c.name);
            Ok(())
        }
    }
}

fn benchmark(game: Option<PathBuf>, model: &str, batch: usize) -> Result<()> {
    let g = match game {
        Some(p) => load(&p)?,
        None => {
            let (_, def, _, _) = generate_valid(&GenSpec::new(Family::Tempo, 1), &ValidationConfig { check_skill: false, ..Default::default() }, 30).map_err(|e| anyhow!(e))?;
            Game::new(def)?
        }
    };
    println!("game {} (hash {:016x})", g.def.name, g.hash64);
    let n = 300u64;
    let mut dec = 0u64;
    let t = Instant::now();
    for s in 0..n {
        let mut st = State::new(&g, s);
        let mut rng = Rng::new(s);
        while !st.is_terminal() {
            let k = st.legal_actions().len();
            st.apply(rng.below(k as u64) as usize);
            dec += 1;
        }
    }
    let el = t.elapsed().as_secs_f64();
    println!("engine: {:.0} games/s, {:.0} decisions/s (state transition + legal-action generation)", n as f64 / el, dec as f64 / el);

    // state clone + apply (search primitive)
    let mut st = State::new(&g, 3);
    let mut rng = Rng::new(3);
    for _ in 0..8 {
        if st.is_terminal() {
            break;
        }
        let k = st.legal_actions().len();
        st.apply(rng.below(k as u64) as usize);
    }
    let t = Instant::now();
    let reps = 20000;
    for i in 0..reps {
        if st.is_terminal() {
            break;
        }
        let mut c = st.clone();
        let k = c.legal_actions().len();
        c.apply(i % k);
    }
    println!("clone+apply: {:.0}/s", reps as f64 / t.elapsed().as_secs_f64());

    let tk = Tokenizer::new(TokenizerConfig::default());
    let t = Instant::now();
    let mut count = 0;
    let mut st = State::new(&g, 5);
    let mut rng = Rng::new(5);
    let mut toks = 0usize;
    while count < 2000 {
        if st.is_terminal() {
            st = State::new(&g, rng.next_u64());
        }
        let p = st.decision_maker().unwrap();
        let o = tk.observe(&mut st, p);
        toks += o.tokens.len();
        count += 1;
        let k = st.legal_actions().len();
        st.apply(rng.below(k as u64) as usize);
    }
    println!("observation: {:.0}/s ({:.1} tokens avg)", count as f64 / t.elapsed().as_secs_f64(), toks as f64 / count as f64);

    let t = Instant::now();
    let mut mcts = Mcts::new(MctsConfig { sims: 100, ..Default::default() }, RolloutEval::new(&g.def));
    let mut st = State::new(&g, 9);
    let mut rng = Rng::new(9);
    for _ in 0..6 {
        let k = st.legal_actions().len();
        st.apply(rng.below(k as u64) as usize);
    }
    let _ = mcts.search(&mut st, &mut rng);
    println!("mcts(100 sims, random rollouts): {:.1} ms/move", t.elapsed().as_secs_f64() * 1000.0);

    backend::model_benchmark(&g, model, batch)
}
