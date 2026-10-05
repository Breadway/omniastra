//! Family "tug": shared tug-of-war track pushed from opposite sides.
//! Strategic structure: spend energy to push vs anchor to protect progress;
//! shared global state with opposite-signed objectives; hidden hands; optional
//! stochastic pushes.

use crate::build::*;
use crate::Variant;
use omnia_dsl::*;
use omnia_engine::Rng;

const DECK: u8 = 0;
const HAND: u8 = 1;
const DISCARD: u8 = 2;
const ENERGY: u8 = 0;
const TRACK: u8 = 0;
const COST: u8 = 0;

fn sign() -> Expr {
    Expr::IfElse(Box::new(cmp(Expr::Seat(PRef::Me), CmpOp::Eq, c(0))), Box::new(c(1)), Box::new(c(-1)))
}

pub fn generate(rng: &mut Rng, variant: Variant) -> GameDef {
    let inverted = variant == Variant::Inverted;
    let t_win = range(rng, 4, 8);
    let energy = range(rng, 2, 4);
    let start_hand = range(rng, 4, 5);
    let hand_cap = range(rng, 6, 8) as u16;
    let ratio = 0.8 + 0.7 * rng.f32();
    let stochastic = chance(rng, 0.4);

    let mut templates: Vec<TemplateDef> = vec![];
    let mut counts: Vec<u16> = vec![];
    for cost in 1..=4 {
        if cost >= 3 && chance(rng, 0.3) {
            continue;
        }
        let base = ((cost as f32) * ratio).round() as i32;
        let p = if inverted { (5 - base).max(1) } else { base.max(1) };
        let mut t = template(&format!("push{}", cost), 0, vec![(COST, cost)]);
        t.hooks.push((0, Effect::ModVar(TRACK, Expr::Mul(Box::new(sign()), Box::new(c(p))))));
        templates.push(t);
        counts.push(range(rng, 3, 6) as u16);
    }
    if stochastic {
        let cost = range(rng, 2, 3);
        let mut t = template("gamble", 0, vec![(COST, cost)]);
        t.hooks.push((0, Effect::ModVar(TRACK, Expr::Mul(Box::new(sign()), Box::new(Expr::Rand(0, cost + 2))))));
        templates.push(t);
        counts.push(range(rng, 2, 4) as u16);
    }
    // anchor: move track toward the centre.
    {
        let q = range(rng, 1, 3);
        let cost = range(rng, 1, 2);
        let mut t = template("anchor", 0, vec![(COST, cost)]);
        let toward = Expr::IfElse(
            Box::new(cmp(Expr::Var(TRACK), CmpOp::Gt, c(0))),
            Box::new(c(-q)),
            Box::new(Expr::IfElse(Box::new(cmp(Expr::Var(TRACK), CmpOp::Lt, c(0))), Box::new(c(q)), Box::new(c(0)))),
        );
        t.hooks.push((0, Effect::ModVar(TRACK, toward)));
        templates.push(t);
        counts.push(range(rng, 2, 4) as u16);
    }
    // cycle: draw.
    {
        let mut t = template("cycle", 0, vec![(COST, 1)]);
        t.hooks.push((0, mv(top_n(me(DECK), 2), me(HAND), Pos::Top)));
        templates.push(t);
        counts.push(range(rng, 2, 4) as u16);
    }
    let player_setup: Vec<SetupEntry> = counts.iter().enumerate().map(|(i, n)| SetupEntry { zone: DECK, template: i as u16, count: *n }).collect();

    let mut play = action("play", 0, vec![1], Timing::Main, seq(vec![Effect::Hook(ORef::Source, 0), mv_one(ORef::Source, me(DISCARD), Pos::Top)]));
    play.source = Some(SourceSpec { zones: vec![me(HAND)], filter: Cond::True });
    play.costs = vec![(ENERGY, attr(ORef::Source, COST))];
    let actions = vec![play, action("end_turn", 1, vec![1], Timing::Main, Effect::EndTurn)];

    let terminal = vec![
        TerminalRule { cond: and(vec![cmp(Expr::Seat(PRef::Me), CmpOp::Eq, c(0)), cmp(Expr::Var(TRACK), CmpOp::Ge, c(t_win))]), result: PlayerResult::Win },
        TerminalRule { cond: and(vec![cmp(Expr::Seat(PRef::Me), CmpOp::Eq, c(1)), cmp(Expr::Var(TRACK), CmpOp::Le, c(-t_win))]), result: PlayerResult::Win },
    ];

    GameDef {
        name: "tug".into(),
        num_players: 2,
        resources: vec![resource("energy", 0, 0, 9)],
        attrs: vec![attr_def("cost", 0, 0, 6)],
        vars: vec![VarDef { name: "track".into(), initial: 0 }],
        zones: vec![
            zone("deck", true, true, Visibility::Hidden, None),
            zone("hand", true, false, Visibility::Private, Some(hand_cap)),
            zone("discard", true, true, Visibility::Public, None),
        ],
        templates,
        hooks: vec!["on_play".into()],
        player_setup,
        shared_setup: vec![],
        setup: for_players(seq(vec![Effect::Shuffle(me(DECK)), mv(top_n(me(DECK), start_hand), me(HAND), Pos::Top)])),
        phases: vec![
            PhaseDef { name: "draw".into(), on_enter: seq(vec![Effect::SetRes(PRef::Me, ENERGY, c(energy)), draw_or_lose(DECK, HAND, 1)]), auto_end: true },
            PhaseDef { name: "main".into(), on_enter: Effect::NoOp, auto_end: false },
        ],
        actions,
        triggers: vec![],
        terminal,
        timeout: Timeout::Draw,
        adjudication: Adjudication::Rules,
        limits: Limits { max_turns: 30, ..Default::default() },
        meta: Meta {
            family: "tug".into(),
            seed: 0,
            tags: tags(&[
                ("information", "hidden"),
                ("resource_pressure", "medium"),
                ("interaction", "high"),
                ("horizon", "medium"),
                ("randomness", if stochastic { "high" } else { "low" }),
                ("tempo_importance", "high"),
                ("resource_conservation", "low"),
                ("engine_building", "none"),
                ("reaction_system", "none"),
                ("tradeoff", "push_vs_anchor"),
            ]),
        },
    }
}
