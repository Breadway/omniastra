//! Turn/phase/stack flow, legal-action enumeration and action application.

use crate::state::*;
use crate::types::*;
use omnia_dsl::*;
use smallvec::SmallVec;

impl State {
    #[inline]
    fn next_player(&self, p: u8) -> u8 {
        (p + 1) % self.game.n()
    }

    // ---- turn structure ---------------------------------------------------

    pub(crate) fn begin_turn(&mut self) {
        let all = self.all_mask();
        self.cause = NO_SEQ;
        self.emit(EventKind::TurnStart, self.active, NONE_OBJ, [NONE_OBJ; 2], 0, 0, self.turn as i32, NO_ZONE, NO_ZONE, 0, all);
        self.fire_delayed(Moment::TurnStart);
        let _ = self.drain_triggers();
        self.check_terminal();
        if self.outcome.is_none() {
            self.enter_phase(0);
        }
    }

    fn fire_delayed(&mut self, at: Moment) {
        let mut due = vec![];
        let mut keep = vec![];
        for d in std::mem::take(&mut self.delayed) {
            if d.at == at && d.fire_turn <= self.turn {
                due.push(d);
            } else {
                keep.push(d);
            }
        }
        self.delayed = keep;
        for d in due {
            let mut c = Ctx::new(d.owner);
            c.source = d.source;
            self.run_top(&d.effect, c);
        }
    }

    fn enter_phase(&mut self, p: u8) {
        self.phase = p;
        let all = self.all_mask();
        self.cause = NO_SEQ;
        self.emit(EventKind::PhaseStart, self.active, NONE_OBJ, [NONE_OBJ; 2], p as i32, 0, 0, NO_ZONE, NO_ZONE, 0, all);
        let g = self.game.clone();
        let ph = &g.def.phases[p as usize];
        self.run_top(&ph.on_enter, Ctx::new(self.active));
        self.check_terminal();
        if self.outcome.is_none() && ph.auto_end && self.transition == Transition::None {
            self.transition = Transition::EndPhase;
        }
    }

    fn do_end_phase(&mut self) {
        let all = self.all_mask();
        self.cause = NO_SEQ;
        self.emit(EventKind::PhaseEnd, self.active, NONE_OBJ, [NONE_OBJ; 2], self.phase as i32, 0, 0, NO_ZONE, NO_ZONE, 0, all);
        let _ = self.drain_triggers();
        self.check_terminal();
        if self.outcome.is_some() {
            return;
        }
        let next = self.phase as usize + 1;
        if next >= self.game.def.phases.len() {
            self.do_end_turn();
        } else {
            self.enter_phase(next as u8);
        }
    }

    fn do_end_turn(&mut self) {
        let all = self.all_mask();
        self.cause = NO_SEQ;
        self.emit(EventKind::TurnEnd, self.active, NONE_OBJ, [NONE_OBJ; 2], 0, 0, self.turn as i32, NO_ZONE, NO_ZONE, 0, all);
        self.fire_delayed(Moment::TurnEnd);
        let _ = self.drain_triggers();
        self.check_terminal();
        if self.outcome.is_some() {
            return;
        }
        self.stack.clear();
        self.active = self.next_player(self.active);
        self.turn += 1;
        if self.turn > self.game.def.limits.max_turns {
            return; // adjudicated in advance()
        }
        self.begin_turn();
    }

    // ---- main loop --------------------------------------------------------

    /// Run automatic game flow until a decision is required or the game ends.
    pub(crate) fn advance(&mut self) {
        let g = self.game.clone();
        let n = g.n();
        let mut guard = 0u32;
        loop {
            guard += 1;
            if guard > 100_000 {
                self.fault.get_or_insert(Fault::NoProgress);
            }
            if self.outcome.is_some() {
                self.legal.clear();
                return;
            }
            if let Some(f) = self.fault {
                self.finish(EndReason::Fault(f), true);
                return;
            }
            if self.turn > g.def.limits.max_turns || self.decisions >= g.def.limits.max_decisions {
                self.finish(EndReason::Timeout, false);
                return;
            }
            match std::mem::replace(&mut self.transition, Transition::None) {
                Transition::EndPhase => {
                    self.do_end_phase();
                    continue;
                }
                Transition::EndTurn => {
                    self.do_end_turn();
                    continue;
                }
                Transition::None => {}
            }
            if !self.stack.is_empty() {
                let p = self.priority;
                let mut acts = self.enumerate(p, Timing::Response);
                if self.fault.is_some() {
                    continue;
                }
                if acts.is_empty() {
                    self.passes += 1;
                    self.priority = self.next_player(p);
                    if self.passes >= n - 1 {
                        self.resolve_top();
                    }
                    continue;
                }
                acts.push(Action { def: PASS_DEF, actor: p, source: NONE_OBJ, targets: Targets::new() });
                self.legal = acts;
                self.decider = p;
                return;
            } else {
                let a = self.active;
                let acts = self.enumerate(a, Timing::Main);
                if self.fault.is_some() {
                    continue;
                }
                if acts.is_empty() {
                    self.transition = Transition::EndPhase;
                    continue;
                }
                self.legal = acts;
                self.decider = a;
                return;
            }
        }
    }

    fn resolve_top(&mut self) {
        let item = match self.stack.pop() {
            Some(i) => i,
            None => return,
        };
        let g = self.game.clone();
        let def = &g.def.actions[item.def as usize];
        let all = self.all_mask();
        self.cause = item.cause_seq;
        if item.cancelled {
            let m = self.obj_mask(item.source);
            self.emit(EventKind::StackCancelled, item.actor, item.source, [NONE_OBJ; 2], item.def as i32, 0, 0, NO_ZONE, NO_ZONE, m, all);
            let _ = self.drain_triggers();
        } else {
            let mut c = Ctx::new(item.actor);
            c.source = item.source;
            c.targets = item.targets.clone();
            // Fizzle only if the source object has been destroyed meanwhile.
            let still_ok = item.source == NONE_OBJ || self.objs[item.source as usize].alive;
            if still_ok {
                let m = self.obj_mask(item.source);
                self.emit(EventKind::StackResolved, item.actor, item.source, [NONE_OBJ; 2], item.def as i32, 0, 0, NO_ZONE, NO_ZONE, m, all);
                self.run_top(&def.effect, c);
            } else {
                let m = self.obj_mask(item.source);
                self.emit(EventKind::StackCancelled, item.actor, item.source, [NONE_OBJ; 2], item.def as i32, 0, 0, NO_ZONE, NO_ZONE, m, all);
                let _ = self.drain_triggers();
            }
        }
        self.check_terminal();
        self.passes = 0;
        self.priority = match self.stack.last() {
            Some(top) => self.next_player(top.actor),
            None => self.active,
        };
    }

    // ---- terminal handling -----------------------------------------------

    pub(crate) fn check_terminal(&mut self) {
        if self.outcome.is_some() {
            return;
        }
        let g = self.game.clone();
        let n = g.n();
        let (mut win, mut lose) = (self.win_mask, self.lose_mask);
        for r in &g.def.terminal {
            for p in 0..n {
                let mut c = Ctx::new(p);
                c.pure = true;
                if self.cond(&r.cond, &mut c) {
                    match r.result {
                        PlayerResult::Win => win |= 1 << p,
                        PlayerResult::Lose => lose |= 1 << p,
                    }
                }
            }
        }
        if win != 0 || lose != 0 || self.draw_flag {
            self.win_mask = win;
            self.lose_mask = lose;
            self.finish(EndReason::Rules, false);
        }
    }

    fn finish(&mut self, reason: EndReason, fault: bool) {
        if self.outcome.is_some() {
            return;
        }
        let g = self.game.clone();
        let n = g.n() as usize;
        let mut pay = [0f32; MAX_PLAYERS];
        let (win, lose) = (self.win_mask, self.lose_mask);
        match reason {
            EndReason::Rules if !fault => {
                let nonlose: Vec<usize> = (0..n).filter(|p| lose & (1 << p) == 0).collect();
                if win != 0 {
                    let winners = (0..n).filter(|p| win & (1 << p) != 0 && lose & (1 << p) == 0).count().max(1) as f32;
                    for p in 0..n {
                        pay[p] = if win & (1 << p) != 0 && lose & (1 << p) == 0 { 1.0 / winners.max(1.0) * if winners > 1.0 { 1.0 } else { 1.0 } } else { -1.0 };
                    }
                    // Shared wins score +1 each (cooperative tie): keep within [-1,1].
                    for p in 0..n {
                        if win & (1 << p) != 0 && lose & (1 << p) == 0 {
                            pay[p] = 1.0;
                        }
                    }
                } else if lose != 0 {
                    for p in 0..n {
                        pay[p] = if lose & (1 << p) != 0 { -1.0 } else if nonlose.len() == 1 { 1.0 } else { 0.0 };
                    }
                }
                // draw_flag with no winners/losers => zeros
            }
            EndReason::Timeout => {
                if let Timeout::ByResource(r) = g.def.timeout {
                    let best = (0..n).map(|p| self.res[p][r as usize]).max().unwrap();
                    let top: Vec<usize> = (0..n).filter(|p| self.res[*p][r as usize] == best).collect();
                    if top.len() == 1 {
                        for p in 0..n {
                            pay[p] = if p == top[0] { 1.0 } else { -1.0 };
                        }
                    }
                }
            }
            _ => {}
        }
        self.outcome = Some(Outcome { payoffs: pay, reason });
        self.legal.clear();
        let all = self.all_mask();
        self.emit(EventKind::GameEnd, NO_PLAYER, NONE_OBJ, [NONE_OBJ; 2], 0, 0, 0, NO_ZONE, NO_ZONE, 0, all);
        self.pending.clear();
    }

    // ---- legal actions ----------------------------------------------------

    pub(crate) fn enumerate(&mut self, actor: u8, timing: Timing) -> Vec<Action> {
        let g = self.game.clone();
        let max_actions = g.def.limits.max_actions as usize;
        let mut out: Vec<Action> = vec![];
        for (di, d) in g.def.actions.iter().enumerate() {
            if d.timing != timing || (!d.phases.is_empty() && !d.phases.contains(&self.phase)) {
                continue;
            }
            let mut c = Ctx::new(actor);
            c.pure = true;
            let sources: SmallVec<[ObjId; 16]> = match &d.source {
                None => SmallVec::from_slice(&[NONE_OBJ]),
                Some(src) => {
                    let sel = Sel::Zones { zones: src.zones.clone(), filter: src.filter.clone(), order: SelOrder::All };
                    self.select(&sel, &mut c)
                }
            };
            for s in sources {
                c.source = s;
                c.targets.clear();
                self.enum_targets(&g, di, d, 0, &mut c, &mut out);
                if out.len() > max_actions {
                    self.fault.get_or_insert(Fault::TooManyActions);
                    return vec![];
                }
            }
        }
        out
    }

    fn enum_targets(&mut self, g: &Game, di: usize, d: &ActionDef, k: usize, c: &mut Ctx, out: &mut Vec<Action>) {
        if k == d.targets.len() {
            // costs
            for (r, x) in &d.costs {
                let amt = self.eval(x, c);
                let min = g.def.resources[*r as usize].min;
                if self.res[c.me as usize][*r as usize] - amt < min || amt < 0 {
                    return;
                }
            }
            if self.cond(&d.require, c) {
                out.push(Action { def: di as u16, actor: c.me, source: c.source, targets: c.targets.clone() });
            }
            return;
        }
        let cands: SmallVec<[Target; 16]> = match &d.targets[k] {
            TargetSpec::Object(sel) => {
                // Candidate may refer to earlier targets via Target(i).
                let taken: SmallVec<[ObjId; 3]> = c.targets.iter().filter_map(|t| if let Target::Obj(o) = t { Some(*o) } else { None }).collect();
                self.select(sel, c).into_iter().filter(|o| !taken.contains(o)).map(Target::Obj).collect()
            }
            TargetSpec::Player(ps) => {
                let n = g.n();
                (0..n).filter(|p| matches!(ps, PSel::All) || *p != c.me).map(Target::Player).collect()
            }
            TargetSpec::Number { lo, hi } => {
                let lo = self.eval(lo, c);
                let hi = self.eval(hi, c).min(lo.saturating_add(15));
                (lo..=hi).map(Target::Num).collect()
            }
        };
        for t in cands {
            c.targets.push(t);
            self.enum_targets(g, di, d, k + 1, c, out);
            c.targets.pop();
            if out.len() > g.def.limits.max_actions as usize {
                return;
            }
        }
    }

    // ---- applying actions -------------------------------------------------

    /// Apply the `idx`-th legal action and advance to the next decision.
    pub fn apply(&mut self, idx: usize) {
        assert!(self.outcome.is_none(), "game is over");
        let a = self.legal[idx].clone();
        self.apply_action(&a);
    }

    pub(crate) fn apply_action(&mut self, a: &Action) {
        self.decisions += 1;
        let g = self.game.clone();
        let all = self.all_mask();
        let n = g.n();
        if a.is_pass() {
            self.cause = NO_SEQ;
            self.passes += 1;
            self.priority = self.next_player(a.actor);
            if self.passes >= n - 1 {
                self.resolve_top();
            }
            self.advance();
            return;
        }
        let d = &g.def.actions[a.def as usize];
        let mut c = Ctx::new(a.actor);
        c.source = a.source;
        c.targets = a.targets.clone();
        self.steps = g.def.limits.max_steps;
        self.cur_gen = 0;
        self.cause = NO_SEQ;
        // Pay costs.
        for (r, x) in &d.costs {
            let amt = self.eval(x, &mut c);
            self.change_res(a.actor, *r, -amt, false, a.actor, a.source);
        }
        // Record the action in history (targets limited to first two objects).
        let mut tg = [NONE_OBJ; 2];
        let mut ti = 0;
        for t in &a.targets {
            if let Target::Obj(o) = t {
                if ti < 2 {
                    tg[ti] = *o;
                    ti += 1;
                }
            }
        }
        // Object identities named by the action are visible to those who could
        // see them before the action resolves, plus the actor.
        let src_mask = if a.source == NONE_OBJ { all } else { self.obj_mask(a.source) | (1 << a.actor) };
        let tgt_mask = tg.iter().filter(|t| **t != NONE_OBJ).fold(all, |acc, t| acc & (self.obj_mask(*t) | (1 << a.actor)));
        let seq = self.emit(EventKind::ActionTaken, a.actor, a.source, tg, a.def as i32, 0, 0, NO_ZONE, NO_ZONE, src_mask & tgt_mask, all);
        self.cause = seq;
        let on_use = d.on_use.clone();
        let _ = self.exec(&on_use, &mut c);
        if d.stack {
            if self.stack.len() as u32 >= g.def.limits.max_stack {
                self.fault.get_or_insert(Fault::StackOverflow);
            } else {
                self.stack.push(StackItem { def: a.def, actor: a.actor, source: a.source, targets: a.targets.clone(), cancelled: false, cause_seq: seq });
                self.passes = 0;
                self.priority = self.next_player(a.actor);
            }
            let _ = self.drain_triggers();
        } else {
            let eff = d.effect.clone();
            let _ = self.exec(&eff, &mut c);
            let _ = self.drain_triggers();
            if d.timing == Timing::Response {
                self.passes = 0;
            }
        }
        self.cause = NO_SEQ;
        self.check_terminal();
        self.advance();
    }
}
