use omnia_dsl::*;

fn gambit() -> GameDef {
    let src = std::fs::read_to_string(format!("{}/../../games/gambit.ron", env!("CARGO_MANIFEST_DIR"))).unwrap();
    GameDef::from_ron(&src).unwrap()
}

fn zone_sel(who: ZWho, zone: ZoneId) -> Sel {
    Sel::Zones { zones: vec![ZRef { who, zone }], filter: Cond::True, order: SelOrder::All }
}

fn errors(g: &GameDef) -> Vec<String> {
    validate(g)
}

fn has(errs: &[String], needle: &str) -> bool {
    errs.iter().any(|e| e.contains(needle))
}

#[test]
fn baseline_game_is_valid() {
    assert_eq!(errors(&gambit()), Vec::<String>::new());
}

#[test]
fn response_action_cannot_target_the_active_players_private_zone() {
    // counter is a Response: the actor need not be the active player.
    let mut g = gambit();
    g.actions[1].targets.push(TargetSpec::Object(zone_sel(ZWho::Active, 1)));
    assert!(has(&errors(&g), "not visible to the actor"), "{:?}", errors(&g));
    // the same target on a Main action is fine (actor == active player)
    let mut g = gambit();
    g.actions[0].targets.push(TargetSpec::Object(zone_sel(ZWho::Active, 1)));
    assert!(!has(&errors(&g), "not visible to the actor"), "{:?}", errors(&g));
}

#[test]
fn source_zones_must_be_visible_to_the_actor() {
    let mut g = gambit();
    g.actions[0].source.as_mut().unwrap().zones = vec![ZRef { who: ZWho::Opp, zone: 1 }];
    assert!(has(&errors(&g), "source selector"), "{:?}", errors(&g));
    let mut g = gambit();
    g.actions[0].source.as_mut().unwrap().zones = vec![ZRef { who: ZWho::Me, zone: 0 }];
    assert!(has(&errors(&g), "source selector"), "{:?}", errors(&g));
}

#[test]
fn owner_of_zone_reference_must_be_in_scope() {
    let mut g = gambit();
    // no targets declared, so Target(0) is out of scope
    g.actions[2].effect = Effect::Move { what: Sel::One(ORef::Last), to: ZRef { who: ZWho::OwnerOf(ORef::Target(0)), zone: 2 }, pos: Pos::Top };
    assert!(has(&errors(&g), "not in scope"), "{:?}", errors(&g));
}

#[test]
fn modifiers_have_no_targets() {
    let mut g = gambit();
    g.templates[0].modifiers.push(ModifierDef { active_in: vec![], affects: Sel::One(ORef::Target(0)), attr: 0, delta: Expr::Const(1) });
    assert!(has(&errors(&g), "not in scope"), "{:?}", errors(&g));
}

#[test]
fn delayed_effects_cannot_reference_targets_or_events() {
    let delayed = |inner: Effect| Effect::Delay { turns: 1, at: Moment::TurnStart, effect: Box::new(inner) };
    let mut g = gambit();
    g.actions[0].targets.push(TargetSpec::Object(zone_sel(ZWho::Me, 1)));
    g.actions[0].effect = delayed(Effect::Destroy(Sel::One(ORef::Target(0))));
    assert!(has(&errors(&g), "not in scope"), "{:?}", errors(&g));
    // the source object survives the delay, so that is fine
    let mut g = gambit();
    g.actions[0].effect = delayed(Effect::Destroy(Sel::One(ORef::Source)));
    assert_eq!(errors(&g), Vec::<String>::new());
}
