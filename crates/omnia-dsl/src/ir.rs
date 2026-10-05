//! OmniDSL canonical intermediate representation.
//!
//! A [`GameDef`] is plain data: the engine interprets it, the generator
//! mutates it, and the ML side derives static descriptors from it. Nothing in
//! here (or anywhere in the generic runtime) knows about a specific game.
//!
//! The effect language is deliberately *not* Turing-complete: loops are
//! bounded (`ForEach` ranges over finite selections, `Repeat` is capped by
//! [`Limits`]), there are no user-defined functions or recursion, and every
//! resolution runs under a step budget.

use serde::{Deserialize, Serialize};

pub type ResId = u8;
pub type AttrId = u8;
pub type VarId = u8;
pub type ZoneId = u8;
pub type TemplateId = u16;
pub type HookId = u8;
pub type PhaseId = u8;
pub type ActionDefId = u16;

/// Hard width limits shared by the engine and the ML tokenizer.
pub const MAX_PLAYERS: usize = 4;
pub const MAX_RESOURCES: usize = 6;
pub const MAX_ATTRS: usize = 12;
pub const MAX_VARS: usize = 4;
pub const MAX_ZONES: usize = 12;
pub const MAX_TEMPLATES: usize = 255;
pub const MAX_ACTION_DEFS: usize = 32;
pub const MAX_TARGETS: usize = 3;

// ---------------------------------------------------------------------------
// Game definition
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GameDef {
    pub name: String,
    pub num_players: u8,
    #[serde(default)]
    pub resources: Vec<ResourceDef>,
    #[serde(default)]
    pub attrs: Vec<AttrDef>,
    #[serde(default)]
    pub vars: Vec<VarDef>,
    pub zones: Vec<ZoneDef>,
    pub templates: Vec<TemplateDef>,
    #[serde(default)]
    pub hooks: Vec<String>,
    /// Per-player starting contents (identical for every player).
    #[serde(default)]
    pub player_setup: Vec<SetupEntry>,
    /// Contents of shared zones.
    #[serde(default)]
    pub shared_setup: Vec<SetupEntry>,
    /// Runs once after objects are created; `Me` is player 0 (use
    /// `ForEachPlayer` to act per player).
    #[serde(default = "Effect::noop")]
    pub setup: Effect,
    pub phases: Vec<PhaseDef>,
    pub actions: Vec<ActionDef>,
    #[serde(default)]
    pub triggers: Vec<TriggerDef>,
    #[serde(default)]
    pub terminal: Vec<TerminalRule>,
    #[serde(default)]
    pub timeout: Timeout,
    /// How final payoffs are assigned. `Random` is an experimental control:
    /// the game ends exactly as it normally would, but the winner is drawn
    /// uniformly at random, so outcomes carry no information about play.
    #[serde(default)]
    pub adjudication: Adjudication,
    #[serde(default)]
    pub limits: Limits,
    /// Free-form generator metadata (never read by the engine).
    #[serde(default)]
    pub meta: Meta,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResourceDef {
    pub name: String,
    pub initial: i32,
    pub min: i32,
    pub max: i32,
    /// If false only the owning player observes the value.
    pub public: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AttrDef {
    pub name: String,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VarDef {
    pub name: String,
    pub initial: i32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// Everyone sees object identities.
    Public,
    /// Only the zone owner sees identities (shared zones: nobody).
    Private,
    /// Nobody sees identities unless explicitly revealed.
    Hidden,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ZoneDef {
    pub name: String,
    pub per_player: bool,
    pub ordered: bool,
    pub vis: Visibility,
    #[serde(default)]
    pub capacity: Option<u16>,
    /// If set, only objects with `kind` in the list may enter.
    #[serde(default)]
    pub allowed_kinds: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TemplateDef {
    pub name: String,
    /// Small categorical type tag (e.g. "creature", "spell" in some game).
    pub kind: u8,
    #[serde(default)]
    pub attrs: Vec<(AttrId, i32)>,
    #[serde(default)]
    pub hooks: Vec<(HookId, Effect)>,
    #[serde(default)]
    pub triggers: Vec<TriggerDef>,
    #[serde(default)]
    pub modifiers: Vec<ModifierDef>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SetupEntry {
    pub zone: ZoneId,
    pub template: TemplateId,
    pub count: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PhaseDef {
    pub name: String,
    #[serde(default = "Effect::noop")]
    pub on_enter: Effect,
    /// Ends immediately after `on_enter` (automatic phases).
    #[serde(default)]
    pub auto_end: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Timing {
    /// Active player, empty stack.
    Main,
    /// Any player holding priority while the stack is non-empty.
    Response,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ActionDef {
    pub name: String,
    /// Categorical class label surfaced to the model.
    pub class: u8,
    /// Empty = any phase.
    #[serde(default)]
    pub phases: Vec<PhaseId>,
    pub timing: Timing,
    #[serde(default)]
    pub source: Option<SourceSpec>,
    #[serde(default)]
    pub targets: Vec<TargetSpec>,
    #[serde(default)]
    pub costs: Vec<(ResId, Expr)>,
    /// Extra legality condition (`Source` / `Target(i)` bound).
    #[serde(default = "Cond::always")]
    pub require: Cond,
    /// Executed immediately after costs are paid (before the action goes on
    /// the stack, if it does) — e.g. moving the played card out of hand.
    #[serde(default = "Effect::noop")]
    pub on_use: Effect,
    pub effect: Effect,
    /// Goes on the stack and opens a response window.
    #[serde(default)]
    pub stack: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceSpec {
    pub zones: Vec<ZRef>,
    /// Candidate object is `Iter`.
    #[serde(default = "Cond::always")]
    pub filter: Cond,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TargetSpec {
    Object(Sel),
    /// Players selected by filter relative to the actor.
    Player(PSel),
    /// Integer parameter in `lo..=hi` (bounds capped at 16 values).
    Number { lo: Expr, hi: Expr },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TriggerDef {
    pub on: EventKind,
    /// For template triggers: zones in which the host object is active
    /// (empty = anywhere).
    #[serde(default)]
    pub active_in: Vec<ZoneId>,
    #[serde(default = "Cond::always")]
    pub cond: Cond,
    pub effect: Effect,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModifierDef {
    /// Zones in which the host object must be for the modifier to apply.
    #[serde(default)]
    pub active_in: Vec<ZoneId>,
    /// Affected objects (`Source` = host, `Iter` = candidate).
    pub affects: Sel,
    pub attr: AttrId,
    pub delta: Expr,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TerminalRule {
    /// Evaluated for each player `p` with `Me = p`.
    pub cond: Cond,
    pub result: PlayerResult,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlayerResult {
    Win,
    Lose,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Timeout {
    Draw,
    /// Highest value of the resource wins; ties draw.
    ByResource(ResId),
}

impl Default for Timeout {
    fn default() -> Self {
        Timeout::Draw
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Adjudication {
    Rules,
    Random,
}

impl Default for Adjudication {
    fn default() -> Self {
        Adjudication::Rules
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Limits {
    pub max_turns: u32,
    pub max_steps: u32,
    pub max_trigger_depth: u32,
    pub max_actions: u32,
    pub max_objects: u32,
    pub max_stack: u32,
    /// Total decisions before the game is adjudicated by the timeout rule.
    pub max_decisions: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_turns: 40,
            max_steps: 4000,
            max_trigger_depth: 24,
            max_actions: 128,
            max_objects: 160,
            max_stack: 8,
            max_decisions: 600,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Meta {
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub tags: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// References and selectors
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PRef {
    Me,
    /// Next player after `Me` in seat order (the opponent in 2p).
    Opp,
    Active,
    Seat(u8),
    OwnerOf(ORef),
    ControllerOf(ORef),
    /// Actor of the event being handled by a trigger.
    EventActor,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ORef {
    Source,
    Target(u8),
    /// Innermost `ForEach` object.
    Iter,
    /// Second-innermost `ForEach` object.
    Outer,
    /// Last object created/copied by the running effect.
    Last,
    /// Subject object of the event being handled by a trigger.
    EventObj,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PSel {
    All,
    /// All players except `Me`.
    Others,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ZWho {
    Me,
    Opp,
    Active,
    /// Owner of an object reference (e.g. `OwnerOf(Source)`).
    OwnerOf(ORef),
    /// Shared (non-per-player) zone.
    Shared,
    /// Every player's instance of the zone.
    Each,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ZRef {
    pub who: ZWho,
    pub zone: ZoneId,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Pos {
    Top,
    Bottom,
    Random,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SelOrder {
    All,
    Top(Box<Expr>),
    Bottom(Box<Expr>),
    Random(Box<Expr>),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Sel {
    One(ORef),
    Zones {
        zones: Vec<ZRef>,
        /// Candidate is `Iter`.
        #[serde(default = "Cond::always")]
        filter: Cond,
        order: SelOrder,
    },
}

// ---------------------------------------------------------------------------
// Expressions and conditions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Expr {
    Const(i32),
    Attr(ORef, AttrId),
    Res(PRef, ResId),
    Var(VarId),
    Count(Sel),
    /// Numeric parameter chosen by the actor (`Target(i)` that is a Number).
    Param(u8),
    Turn,
    /// Seat index of a player.
    Seat(PRef),
    Kind(ORef),
    /// Value carried by the event being handled (delta for resource/attr
    /// changes, otherwise the after-value).
    EventValue,
    /// Slot of the event being handled (resource / attr / action-def / phase id).
    EventSlot,
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    /// Uniform integer in `lo..=hi` using the engine RNG.
    Rand(i32, i32),
    IfElse(Box<Cond>, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Cond {
    True,
    Not(Box<Cond>),
    And(Vec<Cond>),
    Or(Vec<Cond>),
    Cmp(Box<Expr>, CmpOp, Box<Expr>),
    InZone(ORef, ZRef),
    IsTemplate(ORef, TemplateId),
    IsKind(ORef, u8),
    SamePlayer(PRef, PRef),
    Exists(Box<Sel>),
    /// Object is attached to some parent.
    Attached(ORef),
    /// Zone has room (capacity).
    HasRoom(ZRef),
    /// It is the given player's turn.
    IsActive(PRef),
    /// Current phase equals the given one.
    InPhase(PhaseId),
    /// The handled `Moved` event entered / left the given zone (by zone id).
    EventToZone(ZoneId),
    EventFromZone(ZoneId),
}

impl Cond {
    pub fn always() -> Cond {
        Cond::True
    }
}

// ---------------------------------------------------------------------------
// Effects
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EventKind {
    TurnStart,
    TurnEnd,
    PhaseStart,
    PhaseEnd,
    ActionTaken,
    Moved,
    Created,
    Destroyed,
    ResourceChanged,
    AttrChanged,
    Revealed,
    Attached,
    Shuffled,
    ZoneEmpty,
    StackResolved,
    StackCancelled,
    GameEnd,
    Custom(u8),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Moment {
    TurnStart,
    TurnEnd,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Effect {
    Seq(Vec<Effect>),
    If(Cond, Box<Effect>, Box<Effect>),
    /// Binds `Iter` to each selected object in turn (selection snapshotted up front).
    ForEach(Sel, Box<Effect>),
    /// Rebinds `Me` to each selected player in turn.
    ForEachPlayer(PSel, Box<Effect>),
    /// Capped by `Limits::max_steps`.
    Repeat(Expr, Box<Effect>),

    Move { what: Sel, to: ZRef, pos: Pos },
    Create { template: TemplateId, to: ZRef, owner: PRef },
    Copy { what: ORef, to: ZRef },
    Destroy(Sel),
    Transform(Sel, TemplateId),
    SetController(Sel, PRef),

    SetAttr(Sel, AttrId, Expr),
    ModAttr(Sel, AttrId, Expr),
    SetRes(PRef, ResId, Expr),
    /// Signed change.
    ModRes(PRef, ResId, Expr),
    SetVar(VarId, Expr),
    ModVar(VarId, Expr),

    Shuffle(ZRef),
    Reveal(Sel, PSel),
    Hide(Sel),
    Attach(Sel, ORef),
    Detach(Sel),

    Hook(ORef, HookId),
    Emit(u8, Expr),
    Delay { turns: u8, at: Moment, effect: Box<Effect> },
    /// Cancels the item below the one currently resolving (response effects).
    CancelStack,

    Win(PRef),
    Lose(PRef),
    DrawGame,
    EndPhase,
    EndTurn,
    NoOp,
}

impl Effect {
    pub fn noop() -> Effect {
        Effect::NoOp
    }
}
