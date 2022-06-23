use generic_array::typenum::{U4, U6, U8};
use neptune::Poseidon;
use rayon::prelude::*;
use std::hash::Hash;
use std::{fmt, marker::PhantomData};
use string_interner::symbol::{Symbol, SymbolUsize};

use neptune::poseidon::PoseidonConstants;
use once_cell::sync::OnceCell;

use libipld::Cid;

use crate::field::{FWrap, LurkField};
use crate::scalar_store::ScalarContinuation;
use crate::scalar_store::ScalarExpression;
use crate::scalar_store::ScalarStore;
use crate::Num;
use serde::Deserialize;
use serde::Serialize;
use serde::{de, ser};
use serde_repr::{Deserialize_repr, Serialize_repr};

/// Holds the constants needed for poseidon hashing.
#[derive(Debug)]
pub(crate) struct HashConstants<F: LurkField> {
    c4: OnceCell<PoseidonConstants<F, U4>>,
    c6: OnceCell<PoseidonConstants<F, U6>>,
    c8: OnceCell<PoseidonConstants<F, U8>>,
}

impl<F: LurkField> Default for HashConstants<F> {
    fn default() -> Self {
        Self {
            c4: OnceCell::new(),
            c6: OnceCell::new(),
            c8: OnceCell::new(),
        }
    }
}

impl<F: LurkField> HashConstants<F> {
    pub fn c4(&self) -> &PoseidonConstants<F, U4> {
        self.c4.get_or_init(|| PoseidonConstants::new())
    }

    pub fn c6(&self) -> &PoseidonConstants<F, U6> {
        self.c6.get_or_init(|| PoseidonConstants::new())
    }

    pub fn c8(&self) -> &PoseidonConstants<F, U8> {
        self.c8.get_or_init(|| PoseidonConstants::new())
    }
}

type IndexSet<K> = indexmap::IndexSet<K, ahash::RandomState>;

#[derive(Debug)]
struct StringSet(
    string_interner::StringInterner<
        string_interner::backend::BufferBackend<SymbolUsize>,
        ahash::RandomState,
    >,
);

impl Default for StringSet {
    fn default() -> Self {
        StringSet(string_interner::StringInterner::new())
    }
}

#[derive(Debug)]
pub struct Store<F: LurkField> {
    pub(crate) cons_store: IndexSet<(Ptr<F>, Ptr<F>)>,
    pub(crate) comm_store: IndexSet<(FWrap<F>, Ptr<F>)>,

    fun_store: IndexSet<(Ptr<F>, Ptr<F>, Ptr<F>)>,

    sym_store: StringSet,

    // Other sparse storage format without hashing is likely more efficient
    pub(crate) num_store: IndexSet<Num<F>>,

    str_store: StringSet,
    thunk_store: IndexSet<Thunk<F>>,
    call0_store: IndexSet<ContPtr<F>>,
    call_store: IndexSet<(Ptr<F>, Ptr<F>, ContPtr<F>)>,
    call2_store: IndexSet<(Ptr<F>, Ptr<F>, ContPtr<F>)>,
    tail_store: IndexSet<(Ptr<F>, ContPtr<F>)>,
    lookup_store: IndexSet<(Ptr<F>, ContPtr<F>)>,
    unop_store: IndexSet<(Op1, ContPtr<F>)>,
    binop_store: IndexSet<(Op2, Ptr<F>, Ptr<F>, ContPtr<F>)>,
    binop2_store: IndexSet<(Op2, Ptr<F>, ContPtr<F>)>,
    relop_store: IndexSet<(Rel2, Ptr<F>, Ptr<F>, ContPtr<F>)>,
    relop2_store: IndexSet<(Rel2, Ptr<F>, ContPtr<F>)>,
    if_store: IndexSet<(Ptr<F>, ContPtr<F>)>,
    let_store: IndexSet<(Ptr<F>, Ptr<F>, Ptr<F>, ContPtr<F>)>,
    let_rec_store: IndexSet<(Ptr<F>, Ptr<F>, Ptr<F>, ContPtr<F>)>,
    emit_store: IndexSet<ContPtr<F>>,

    opaque_map: dashmap::DashMap<Ptr<F>, ScalarPtr<F>>,
    /// Holds a mapping of ScalarPtr -> Ptr for reverse lookups
    pub(crate) scalar_ptr_map: dashmap::DashMap<ScalarPtr<F>, Ptr<F>, ahash::RandomState>,
    /// Holds a mapping of ScalarPtr -> ContPtr<F> for reverse lookups
    scalar_ptr_cont_map: dashmap::DashMap<ScalarContPtr<F>, ContPtr<F>, ahash::RandomState>,

    /// Caches poseidon hashes
    poseidon_cache: PoseidonCache<F>,
    /// Contains Ptrs which have not yet been hydrated.
    dehydrated: Vec<Ptr<F>>,
    dehydrated_cont: Vec<ContPtr<F>>,
    opaque_raw_ptr_count: usize,
}

#[derive(Default, Debug)]
struct PoseidonCache<F: LurkField> {
    a4: dashmap::DashMap<CacheKey<F, 4>, F, ahash::RandomState>,
    a6: dashmap::DashMap<CacheKey<F, 6>, F, ahash::RandomState>,
    a8: dashmap::DashMap<CacheKey<F, 8>, F, ahash::RandomState>,

    constants: HashConstants<F>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct CacheKey<F: LurkField, const N: usize>([F; N]);

#[allow(clippy::derive_hash_xor_eq)]
impl<F: LurkField, const N: usize> Hash for CacheKey<F, N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for el in &self.0 {
            el.to_repr().as_ref().hash(state);
        }
    }
}

impl<F: LurkField> PoseidonCache<F> {
    fn hash4(&self, preimage: &[F; 4]) -> F {
        let hash = self
            .a4
            .entry(CacheKey(*preimage))
            .or_insert_with(|| Poseidon::new_with_preimage(preimage, self.constants.c4()).hash());

        *hash
    }

    fn hash6(&self, preimage: &[F; 6]) -> F {
        let hash = self
            .a6
            .entry(CacheKey(*preimage))
            .or_insert_with(|| Poseidon::new_with_preimage(preimage, self.constants.c6()).hash());
        *hash
    }

    fn hash8(&self, preimage: &[F; 8]) -> F {
        let hash = self
            .a8
            .entry(CacheKey(*preimage))
            .or_insert_with(|| Poseidon::new_with_preimage(preimage, self.constants.c8()).hash());
        *hash
    }
}

pub trait Object<F: LurkField>: fmt::Debug + Copy + Clone + PartialEq {
    type Pointer: Pointer<F>;
}

pub trait Pointer<F: LurkField + From<u64>>: fmt::Debug + Copy + Clone + PartialEq + Hash {
    type Tag: Into<u64>;
    type ScalarPointer: ScalarPointer<F>;

    fn tag(&self) -> Self::Tag;
    fn tag_field(&self) -> F {
        F::from(self.tag().into())
    }
}

pub trait ScalarPointer<F: LurkField>: fmt::Debug + Copy + Clone + PartialEq + Hash {
    fn from_parts(tag: F, value: F) -> Self;
    fn tag(&self) -> &F;
    fn value(&self) -> &F;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Ptr<F: LurkField>(Tag, RawPtr<F>);

#[allow(clippy::derive_hash_xor_eq)]
impl<F: LurkField> Hash for Ptr<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl<F: LurkField> Ptr<F> {
    pub fn is_nil(&self) -> bool {
        matches!(self.0, Tag::Nil)
    }

    pub fn is_fun(&self) -> bool {
        matches!(self.0, Tag::Fun)
    }

    pub fn is_opaque(&self) -> bool {
        self.1.is_opaque()
    }
}

impl<F: LurkField> From<char> for Ptr<F> {
    fn from(c: char) -> Self {
        Self(Tag::Char, RawPtr::new(u32::from(c) as usize))
    }
}

impl<F: LurkField> Pointer<F> for Ptr<F> {
    type Tag = Tag;
    type ScalarPointer = ScalarPtr<F>;

    fn tag(&self) -> Tag {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarPtr<F: LurkField>(F, F);

impl<F: LurkField> Copy for ScalarPtr<F> {}

impl<F: LurkField> PartialOrd for ScalarPtr<F> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        (self.0.to_repr().as_ref(), self.1.to_repr().as_ref())
            .partial_cmp(&(other.0.to_repr().as_ref(), other.1.to_repr().as_ref()))
    }
}

impl<F: LurkField> Ord for ScalarPtr<F> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.0.to_repr().as_ref(), self.1.to_repr().as_ref())
            .cmp(&(other.0.to_repr().as_ref(), other.1.to_repr().as_ref()))
    }
}

impl<F: LurkField> Serialize for ScalarPtr<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use ser::Error;
        let cid = F::to_cid(self.0, self.1)
            .ok_or_else(|| S::Error::custom("expected validly tagged ScalarPtr".to_string()))?;
        cid.serialize(serializer)
    }
}

impl<'de, F: LurkField> Deserialize<'de> for ScalarPtr<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use de::Error;
        let cid = Cid::deserialize(deserializer)?;
        let (tag, dig) = F::from_cid(cid).ok_or_else(|| {
            D::Error::custom(format!("expected ScalarPtr encoded as Cid, got {}", cid))
        })?;
        Ok(ScalarPtr::from_parts(tag, dig))
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<F: LurkField> Hash for ScalarPtr<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_repr().as_ref().hash(state);
        self.1.to_repr().as_ref().hash(state);
    }
}

impl<F: LurkField> ScalarPointer<F> for ScalarPtr<F> {
    fn from_parts(tag: F, value: F) -> Self {
        ScalarPtr(tag, value)
    }

    fn tag(&self) -> &F {
        &self.0
    }

    fn value(&self) -> &F {
        &self.1
    }
}

pub trait IntoHashComponents<F: LurkField> {
    fn into_hash_components(self) -> [F; 2];
}

impl<F: LurkField> IntoHashComponents<F> for [F; 2] {
    fn into_hash_components(self) -> [F; 2] {
        self
    }
}

impl<F: LurkField> IntoHashComponents<F> for ScalarPtr<F> {
    fn into_hash_components(self) -> [F; 2] {
        [self.0, self.1]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarContPtr<F: LurkField>(F, F);

impl<F: LurkField> Copy for ScalarContPtr<F> {}

impl<F: LurkField> PartialOrd for ScalarContPtr<F> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        (self.0.to_repr().as_ref(), self.1.to_repr().as_ref())
            .partial_cmp(&(other.0.to_repr().as_ref(), other.1.to_repr().as_ref()))
    }
}

impl<F: LurkField> Ord for ScalarContPtr<F> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.0.to_repr().as_ref(), self.1.to_repr().as_ref())
            .cmp(&(other.0.to_repr().as_ref(), other.1.to_repr().as_ref()))
    }
}

impl<F: LurkField> Serialize for ScalarContPtr<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use ser::Error;
        let cid = F::to_cid(self.0, self.1)
            .ok_or_else(|| S::Error::custom("expected validly tagged ScalarContPtr".to_string()))?;
        cid.serialize(serializer)
    }
}

impl<'de, F: LurkField> Deserialize<'de> for ScalarContPtr<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use de::Error;
        let cid = Cid::deserialize(deserializer)?;
        let (tag, dig) = F::from_cid(cid).ok_or_else(|| {
            D::Error::custom(format!(
                "expected ScalarContPtr encoded as Cid, got {}",
                cid
            ))
        })?;
        Ok(ScalarContPtr::from_parts(tag, dig))
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<F: LurkField> Hash for ScalarContPtr<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_repr().as_ref().hash(state);
        self.1.to_repr().as_ref().hash(state);
    }
}

impl<F: LurkField> ScalarPointer<F> for ScalarContPtr<F> {
    fn from_parts(tag: F, value: F) -> Self {
        ScalarContPtr(tag, value)
    }
    fn tag(&self) -> &F {
        &self.0
    }

    fn value(&self) -> &F {
        &self.1
    }
}

impl<F: LurkField> IntoHashComponents<F> for ScalarContPtr<F> {
    fn into_hash_components(self) -> [F; 2] {
        [self.0, self.1]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ContPtr<F: LurkField>(ContTag, RawPtr<F>);

#[allow(clippy::derive_hash_xor_eq)]
impl<F: LurkField> Hash for ContPtr<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl<F: LurkField> Pointer<F> for ContPtr<F> {
    type Tag = ContTag;
    type ScalarPointer = ScalarContPtr<F>;

    fn tag(&self) -> Self::Tag {
        self.0
    }
}

impl<F: LurkField> ContPtr<F> {
    pub fn is_error(&self) -> bool {
        matches!(self.0, ContTag::Error)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
// If .0 is negative, RawPtr is opaque. This lets us retain the efficiency and structure of the current implementation.
// It cuts the local store's address space in half, which is likely not an issue. This representation does not affect
// external data, so if we want to change it in the future, we can do so without a change of defined behavior.
pub struct RawPtr<F: LurkField>(isize, PhantomData<F>);

impl<F: LurkField> RawPtr<F> {
    fn new(p: usize) -> Self {
        assert!(p < isize::MAX as usize);
        RawPtr(p as isize, Default::default())
    }

    fn is_opaque(&self) -> bool {
        self.0 < 0
    }

    pub fn idx(&self) -> usize {
        self.0.unsigned_abs() as usize
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<F: LurkField> Hash for RawPtr<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

// Expressions, Continuations, Op1, Op2, and Rel2 occupy the same namespace in
// their encoding.
// As a 16bit integer their representation is as follows
//    [typ] [value       ]
// 0b 0000_ 0000_0000_0000
//
// where typ is
// - `0b0000` for Tag
// - `0b0001` for ContTag
// - `0b0010` for Op1
// - `0b0011` for Op2
// - `0b0100` for Rel2

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Expression<'a, F: LurkField> {
    Nil,
    Cons(Ptr<F>, Ptr<F>),
    Comm(F, Ptr<F>),
    Sym(&'a str),
    /// arg, body, closed env
    Fun(Ptr<F>, Ptr<F>, Ptr<F>),
    Num(Num<F>),
    Str(&'a str),
    Thunk(Thunk<F>),
    Opaque(Ptr<F>),
    Char(char),
}

impl<F: LurkField> Object<F> for Expression<'_, F> {
    type Pointer = Ptr<F>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thunk<F: LurkField> {
    pub(crate) value: Ptr<F>,
    pub(crate) continuation: ContPtr<F>,
}

#[allow(clippy::derive_hash_xor_eq)]
impl<F: LurkField> Hash for Thunk<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
        self.continuation.hash(state);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Continuation<F: LurkField> {
    Outermost,
    Call0 {
        continuation: ContPtr<F>,
    },
    Call {
        unevaled_arg: Ptr<F>,
        saved_env: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Call2 {
        function: Ptr<F>,
        saved_env: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Tail {
        saved_env: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Error,
    Lookup {
        saved_env: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Unop {
        operator: Op1,
        continuation: ContPtr<F>,
    },
    Binop {
        operator: Op2,
        saved_env: Ptr<F>,
        unevaled_args: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Binop2 {
        operator: Op2,
        evaled_arg: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Relop {
        operator: Rel2,
        saved_env: Ptr<F>,
        unevaled_args: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Relop2 {
        operator: Rel2,
        evaled_arg: Ptr<F>,
        continuation: ContPtr<F>,
    },
    If {
        unevaled_args: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Let {
        var: Ptr<F>,
        body: Ptr<F>,
        saved_env: Ptr<F>,
        continuation: ContPtr<F>,
    },
    LetRec {
        var: Ptr<F>,
        saved_env: Ptr<F>,
        body: Ptr<F>,
        continuation: ContPtr<F>,
    },
    Emit {
        continuation: ContPtr<F>,
    },
    Dummy,
    Terminal,
}

impl<F: LurkField> Object<F> for Continuation<F> {
    type Pointer = ContPtr<F>;
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Hash, Serialize_repr, Deserialize_repr)]
#[repr(u16)]
pub enum Op1 {
    Car = 0b0010_0000_0000_0000,
    Cdr,
    Atom,
    Emit,
    Open,
    Secret,
    Commit,
    Num,
    Comm,
}

impl fmt::Display for Op1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op1::Car => write!(f, "Car"),
            Op1::Cdr => write!(f, "Cdr"),
            Op1::Atom => write!(f, "Atom"),
            Op1::Emit => write!(f, "Emit"),
            Op1::Open => write!(f, "Open"),
            Op1::Secret => write!(f, "Secret"),
            Op1::Commit => write!(f, "Commit"),
            Op1::Num => write!(f, "Num"),
            Op1::Comm => write!(f, "Comm"),
        }
    }
}

impl Op1 {
    pub fn from_u16(x: u16) -> Option<Self> {
        match x {
            x if x == Op1::Car as u16 => Some(Op1::Car),
            x if x == Op1::Cdr as u16 => Some(Op1::Cdr),
            x if x == Op1::Atom as u16 => Some(Op1::Atom),
            x if x == Op1::Emit as u16 => Some(Op1::Emit),
            x if x == Op1::Open as u16 => Some(Op1::Open),
            x if x == Op1::Secret as u16 => Some(Op1::Secret),
            x if x == Op1::Commit as u16 => Some(Op1::Commit),
            x if x == Op1::Num as u16 => Some(Op1::Num),
            x if x == Op1::Comm as u16 => Some(Op1::Comm),
            _ => None,
        }
    }

    pub fn as_field<F: From<u64> + ff::Field>(&self) -> F {
        F::from(*self as u64)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Hash, Serialize_repr, Deserialize_repr)]
#[repr(u16)]
pub enum Op2 {
    Sum = 0b0011_0000_0000_0000,
    Diff,
    Product,
    Quotient,
    Cons,
    Begin,
    Hide,
}

impl Op2 {
    pub fn from_u16(x: u16) -> Option<Self> {
        match x {
            x if x == Op2::Sum as u16 => Some(Op2::Sum),
            x if x == Op2::Diff as u16 => Some(Op2::Diff),
            x if x == Op2::Product as u16 => Some(Op2::Product),
            x if x == Op2::Quotient as u16 => Some(Op2::Quotient),
            x if x == Op2::Cons as u16 => Some(Op2::Cons),
            x if x == Op2::Begin as u16 => Some(Op2::Begin),
            x if x == Op2::Hide as u16 => Some(Op2::Hide),
            _ => None,
        }
    }
    pub fn as_field<F: From<u64> + ff::Field>(&self) -> F {
        F::from(*self as u64)
    }
}

impl fmt::Display for Op2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op2::Sum => write!(f, "Sum"),
            Op2::Diff => write!(f, "Diff"),
            Op2::Product => write!(f, "Product"),
            Op2::Quotient => write!(f, "Quotient"),
            Op2::Cons => write!(f, "Cons"),
            Op2::Begin => write!(f, "Begin"),
            Op2::Hide => write!(f, "Hide"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Hash, Serialize_repr, Deserialize_repr)]
#[repr(u16)]
pub enum Rel2 {
    Equal = 0b0100_0000_0000_0000,
    NumEqual,
}

impl Rel2 {
    pub fn from_u16(x: u16) -> Option<Self> {
        match x {
            x if x == Rel2::Equal as u16 => Some(Rel2::Equal),
            x if x == Rel2::NumEqual as u16 => Some(Rel2::NumEqual),
            _ => None,
        }
    }
    pub fn as_field<F: From<u64> + ff::Field>(&self) -> F {
        F::from(*self as u64)
    }
}

impl fmt::Display for Rel2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rel2::Equal => write!(f, "Equal"),
            Rel2::NumEqual => write!(f, "NumEqual"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize_repr, Deserialize_repr)]
#[repr(u16)]
pub enum Tag {
    Nil = 0b0000_0000_0000_0000,
    Cons,
    Sym,
    Fun,
    Num,
    Thunk,
    Str,
    Char,
    Comm,
}

impl From<Tag> for u64 {
    fn from(t: Tag) -> Self {
        t as u64
    }
}

impl Tag {
    pub fn from_field<F: From<u64> + ff::Field>(f: F) -> Option<Self> {
        match f {
            f if f == Tag::Nil.as_field() => Some(Tag::Nil),
            f if f == Tag::Cons.as_field() => Some(Tag::Cons),
            f if f == Tag::Sym.as_field() => Some(Tag::Sym),
            f if f == Tag::Fun.as_field() => Some(Tag::Fun),
            f if f == Tag::Thunk.as_field() => Some(Tag::Thunk),
            f if f == Tag::Num.as_field() => Some(Tag::Num),
            f if f == Tag::Str.as_field() => Some(Tag::Str),
            _ => None,
        }
    }

    pub fn as_field<F: From<u64> + ff::Field>(&self) -> F {
        F::from(*self as u64)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum ContTag {
    Outermost = 0b0001_0000_0000_0000,
    Call0,
    Call,
    Call2,
    Tail,
    Error,
    Lookup,
    Unop,
    Binop,
    Binop2,
    Relop,
    Relop2,
    If,
    Let,
    LetRec,
    Dummy,
    Terminal,
    Emit,
}

impl From<ContTag> for u64 {
    fn from(t: ContTag) -> Self {
        t as u64
    }
}

impl ContTag {
    pub fn from_field<F: From<u64> + ff::Field>(f: F) -> Option<Self> {
        match f {
            f if f == ContTag::Outermost.as_field() => Some(ContTag::Outermost),
            f if f == ContTag::Call0.as_field() => Some(ContTag::Call0),
            f if f == ContTag::Call.as_field() => Some(ContTag::Call),
            f if f == ContTag::Call2.as_field() => Some(ContTag::Call2),
            f if f == ContTag::Tail.as_field() => Some(ContTag::Tail),
            f if f == ContTag::Error.as_field() => Some(ContTag::Error),
            f if f == ContTag::Lookup.as_field() => Some(ContTag::Lookup),
            f if f == ContTag::Unop.as_field() => Some(ContTag::Unop),
            f if f == ContTag::Binop.as_field() => Some(ContTag::Binop),
            f if f == ContTag::Relop.as_field() => Some(ContTag::Relop),
            f if f == ContTag::If.as_field() => Some(ContTag::If),
            f if f == ContTag::Let.as_field() => Some(ContTag::Let),
            f if f == ContTag::LetRec.as_field() => Some(ContTag::LetRec),
            f if f == ContTag::Dummy.as_field() => Some(ContTag::Dummy),
            f if f == ContTag::Terminal.as_field() => Some(ContTag::Terminal),
            f if f == ContTag::Emit.as_field() => Some(ContTag::Emit),
            _ => None,
        }
    }
    pub fn as_field<F: From<u64> + ff::Field>(&self) -> F {
        F::from(*self as u64)
    }
}

impl<F: LurkField> Default for Store<F> {
    fn default() -> Self {
        let mut store = Store {
            cons_store: Default::default(),
            comm_store: Default::default(),
            sym_store: Default::default(),
            num_store: Default::default(),
            fun_store: Default::default(),
            str_store: Default::default(),
            thunk_store: Default::default(),
            call0_store: Default::default(),
            call_store: Default::default(),
            call2_store: Default::default(),
            tail_store: Default::default(),
            lookup_store: Default::default(),
            unop_store: Default::default(),
            binop_store: Default::default(),
            binop2_store: Default::default(),
            relop_store: Default::default(),
            relop2_store: Default::default(),
            if_store: Default::default(),
            let_store: Default::default(),
            let_rec_store: Default::default(),
            emit_store: Default::default(),
            opaque_map: Default::default(),
            scalar_ptr_map: Default::default(),
            scalar_ptr_cont_map: Default::default(),
            poseidon_cache: Default::default(),
            dehydrated: Default::default(),
            dehydrated_cont: Default::default(),
            opaque_raw_ptr_count: 0,
        };

        // insert some well known symbols
        for sym in &[
            "nil",
            "t",
            "quote",
            "lambda",
            "_",
            "let",
            "letrec",
            "begin",
            "cons",
            "car",
            "cdr",
            "atom",
            "emit",
            "+",
            "-",
            "*",
            "/",
            "=",
            "eq",
            "current-env",
            "if",
            "terminal",
            "dummy",
            "outermost",
            "error",
        ] {
            store.sym(sym);
        }

        store
    }
}

/// These methods provide a more ergonomic means of constructing and manipulating Lurk data.
/// They can be thought of as a minimal DSL for working with Lurk data in Rust code.
/// Prefer these methods when constructing literal data or assembling program fragments in
/// tests or during evaluation, etc.
impl<F: LurkField> Store<F> {
    pub fn nil(&mut self) -> Ptr<F> {
        self.intern_nil()
    }

    pub fn t(&mut self) -> Ptr<F> {
        self.sym("T")
    }

    pub fn cons(&mut self, car: Ptr<F>, cdr: Ptr<F>) -> Ptr<F> {
        self.intern_cons(car, cdr)
    }

    pub fn hide(&mut self, secret: F, payload: Ptr<F>) -> Ptr<F> {
        self.intern_comm(secret, payload)
    }

    pub fn list(&mut self, elts: &[Ptr<F>]) -> Ptr<F> {
        self.intern_list(elts)
    }

    pub fn num<T: Into<Num<F>>>(&mut self, num: T) -> Ptr<F> {
        self.intern_num(num)
    }

    pub fn str<T: AsRef<str>>(&mut self, name: T) -> Ptr<F> {
        self.intern_str(name)
    }

    pub fn sym<T: AsRef<str>>(&mut self, name: T) -> Ptr<F> {
        self.intern_sym_with_case_conversion(name)
    }

    pub fn car(&self, expr: &Ptr<F>) -> Ptr<F> {
        self.car_cdr(expr).0
    }

    pub fn cdr(&self, expr: &Ptr<F>) -> Ptr<F> {
        self.car_cdr(expr).1
    }

    pub(crate) fn poseidon_constants(&self) -> &HashConstants<F> {
        &self.poseidon_cache.constants
    }
}

impl<F: LurkField> Store<F> {
    pub fn new() -> Self {
        Store::default()
    }

    pub fn intern_nil(&mut self) -> Ptr<F> {
        self.sym("nil")
    }

    pub fn get_nil(&self) -> Ptr<F> {
        self.get_sym("nil", true).expect("missing NIL")
    }

    pub fn get_begin(&self) -> Ptr<F> {
        self.get_sym("begin", true).expect("missing BEGIN")
    }

    pub fn get_quote(&self) -> Ptr<F> {
        self.get_sym("quote", true).expect("missing QUOTE")
    }

    pub fn get_t(&self) -> Ptr<F> {
        self.get_sym("t", true).expect("missing T")
    }

    pub fn intern_cons(&mut self, car: Ptr<F>, cdr: Ptr<F>) -> Ptr<F> {
        if car.is_opaque() || cdr.is_opaque() {
            self.hash_expr(&car);
            self.hash_expr(&cdr);
        }
        if (car.tag(), cdr.tag()) == (Tag::Char, Tag::Str) {
            let (c, s) = (
                self.fetch_char(&car).unwrap(),
                self.fetch_str(&cdr).unwrap(),
            );
            let new_str = format!("{}{}", c, s);
            return self.intern_str(&new_str);
        };

        let (p, inserted) = self.cons_store.insert_full((car, cdr));
        let ptr = Ptr(Tag::Cons, RawPtr::new(p));
        if inserted {
            self.dehydrated.push(ptr);
        }
        ptr
    }
    pub fn intern_comm(&mut self, secret: F, payload: Ptr<F>) -> Ptr<F> {
        if payload.is_opaque() {
            self.hash_expr(&payload);
        }
        let (p, inserted) = self.comm_store.insert_full((FWrap(secret), payload));

        let ptr = Ptr(Tag::Comm, RawPtr::new(p));
        if inserted {
            self.dehydrated.push(ptr);
        }
        ptr
    }

    // Intern a potentially-opaque value. If the corresponding value is already known to the store,
    // return the known value.
    fn intern_maybe_opaque(&mut self, tag: Tag, hash: F) -> Ptr<F> {
        self.intern_opaque_aux(tag, hash, true)
    }

    // Intern an opaque value. If the corresponding non-opaque value is already known to the store,
    // return an opaque one anyway.
    fn intern_opaque(&mut self, tag: Tag, hash: F) -> Ptr<F> {
        self.intern_opaque_aux(tag, hash, false)
    }

    // Intern a potentially-opaque value. If the corresponding non-opaque value is already known to the store, and
    // `return_non_opaque_if_existing` is true, return the known value.
    fn intern_opaque_aux(
        &mut self,
        tag: Tag,
        hash: F,
        return_non_opaque_if_existing: bool,
    ) -> Ptr<F> {
        self.hydrate_scalar_cache();
        let scalar_ptr = ScalarPtr::from_parts(tag.as_field(), hash);

        // Scope the first immutable borrow.
        {
            let ptr = self.scalar_ptr_map.get(&scalar_ptr);
            if let Some(p) = ptr {
                if return_non_opaque_if_existing || p.is_opaque() {
                    return *p;
                }
            }
        }

        let ptr = Ptr(tag, self.new_opaque_raw_ptr());
        // Always insert. Key is unique because of newly allocated opaque raw_ptr.
        self.opaque_map.insert(ptr, scalar_ptr);
        ptr
    }

    pub fn intern_scalar_cont_ptr(
        &mut self,
        ptr: ScalarContPtr<F>,
        scalar_store: &ScalarStore<F>,
    ) -> Option<ContPtr<F>> {
        let tag: ContTag = ContTag::from_field(*ptr.tag())?;
        let cont = scalar_store.get_cont(&ptr);
        use ScalarContinuation::*;
        match (tag, cont) {
            (ContTag::Outermost, Some(Outermost)) => Some(self.intern_cont_outermost()),
            (
                ContTag::Call,
                Some(Call {
                    unevaled_arg,
                    saved_env,
                    continuation,
                }),
            ) => {
                let arg = self.intern_scalar_ptr(*unevaled_arg, scalar_store)?;
                let env = self.intern_scalar_ptr(*saved_env, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_call(arg, env, cont))
            }
            (
                ContTag::Call2,
                Some(Call2 {
                    function,
                    saved_env,
                    continuation,
                }),
            ) => {
                let fun = self.intern_scalar_ptr(*function, scalar_store)?;
                let env = self.intern_scalar_ptr(*saved_env, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_call2(fun, env, cont))
            }
            (
                ContTag::Tail,
                Some(Tail {
                    saved_env,
                    continuation,
                }),
            ) => {
                let env = self.intern_scalar_ptr(*saved_env, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_tail(env, cont))
            }
            (ContTag::Error, Some(Error)) => Some(self.intern_cont_error()),
            (
                ContTag::Lookup,
                Some(Lookup {
                    saved_env,
                    continuation,
                }),
            ) => {
                let env = self.intern_scalar_ptr(*saved_env, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_lookup(env, cont))
            }
            (
                ContTag::Unop,
                Some(Unop {
                    operator,
                    continuation,
                }),
            ) => {
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_unop(*operator, cont))
            }
            (
                ContTag::Binop,
                Some(Binop {
                    operator,
                    saved_env,
                    unevaled_args,
                    continuation,
                }),
            ) => {
                let env = self.intern_scalar_ptr(*saved_env, scalar_store)?;
                let args = self.intern_scalar_ptr(*unevaled_args, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_binop(*operator, env, args, cont))
            }
            (
                ContTag::Binop2,
                Some(Binop2 {
                    operator,
                    evaled_arg,
                    continuation,
                }),
            ) => {
                let arg = self.intern_scalar_ptr(*evaled_arg, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_binop2(*operator, arg, cont))
            }
            (
                ContTag::Relop,
                Some(Relop {
                    operator,
                    saved_env,
                    unevaled_args,
                    continuation,
                }),
            ) => {
                let env = self.intern_scalar_ptr(*saved_env, scalar_store)?;
                let args = self.intern_scalar_ptr(*unevaled_args, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_relop(*operator, env, args, cont))
            }
            (
                ContTag::Relop2,
                Some(Relop2 {
                    operator,
                    evaled_arg,
                    continuation,
                }),
            ) => {
                let arg = self.intern_scalar_ptr(*evaled_arg, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_relop2(*operator, arg, cont))
            }
            (
                ContTag::If,
                Some(If {
                    unevaled_args,
                    continuation,
                }),
            ) => {
                let args = self.intern_scalar_ptr(*unevaled_args, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_if(args, cont))
            }
            (
                ContTag::Let,
                Some(Let {
                    var,
                    body,
                    saved_env,
                    continuation,
                }),
            ) => {
                let var = self.intern_scalar_ptr(*var, scalar_store)?;
                let body = self.intern_scalar_ptr(*body, scalar_store)?;
                let env = self.intern_scalar_ptr(*saved_env, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_let(var, body, env, cont))
            }
            (
                ContTag::LetRec,
                Some(LetRec {
                    var,
                    body,
                    saved_env,
                    continuation,
                }),
            ) => {
                let var = self.intern_scalar_ptr(*var, scalar_store)?;
                let body = self.intern_scalar_ptr(*body, scalar_store)?;
                let env = self.intern_scalar_ptr(*saved_env, scalar_store)?;
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_let(var, body, env, cont))
            }
            (ContTag::Emit, Some(Emit { continuation })) => {
                let cont = self.intern_scalar_cont_ptr(*continuation, scalar_store)?;
                Some(self.intern_cont_emit(cont))
            }
            (ContTag::Dummy, Some(Dummy)) => Some(self.intern_cont_dummy()),
            (ContTag::Terminal, Some(Terminal)) => Some(self.intern_cont_terminal()),
            _ => None,
        }
    }

    pub fn intern_scalar_ptr(
        &mut self,
        ptr: ScalarPtr<F>,
        scalar_store: &ScalarStore<F>,
    ) -> Option<Ptr<F>> {
        let tag: Tag = Tag::from_field(*ptr.tag())?;
        let expr = scalar_store.get_expr(&ptr);
        use ScalarExpression::*;
        match (tag, expr) {
            (Tag::Nil, Some(Nil)) => Some(self.intern_nil()),
            (Tag::Cons, Some(Cons(car, cdr))) => {
                let car = self.intern_scalar_ptr(*car, scalar_store)?;
                let cdr = self.intern_scalar_ptr(*cdr, scalar_store)?;
                Some(self.intern_cons(car, cdr))
            }
            (Tag::Str, Some(Str(s))) => Some(self.intern_str(s)),
            (Tag::Sym, Some(Sym(s))) => Some(self.intern_sym(s)),
            (Tag::Num, Some(Num(x))) => Some(self.intern_num(crate::Num::Scalar(*x))),
            (Tag::Thunk, Some(Thunk(t))) => {
                let value = self.intern_scalar_ptr(t.value, scalar_store)?;
                let continuation = self.intern_scalar_cont_ptr(t.continuation, scalar_store)?;
                Some(self.intern_thunk(crate::store::Thunk {
                    value,
                    continuation,
                }))
            }
            (
                Tag::Fun,
                Some(Fun {
                    arg,
                    body,
                    closed_env,
                }),
            ) => {
                let arg = self.intern_scalar_ptr(*arg, scalar_store)?;
                let body = self.intern_scalar_ptr(*body, scalar_store)?;
                let env = self.intern_scalar_ptr(*closed_env, scalar_store)?;
                Some(self.intern_fun(arg, body, env))
            }
            (tag, None) => Some(self.intern_maybe_opaque(tag, ptr.1)),
            _ => None,
        }
    }

    pub fn intern_maybe_opaque_fun(&mut self, hash: F) -> Ptr<F> {
        self.intern_maybe_opaque(Tag::Fun, hash)
    }

    pub fn intern_maybe_opaque_sym(&mut self, hash: F) -> Ptr<F> {
        self.intern_maybe_opaque(Tag::Sym, hash)
    }

    pub fn intern_maybe_opaque_cons(&mut self, hash: F) -> Ptr<F> {
        self.intern_maybe_opaque(Tag::Cons, hash)
    }

    pub fn intern_maybe_opaque_comm(&mut self, hash: F) -> Ptr<F> {
        self.intern_maybe_opaque(Tag::Comm, hash)
    }

    pub fn intern_opaque_fun(&mut self, hash: F) -> Ptr<F> {
        self.intern_opaque(Tag::Fun, hash)
    }

    pub fn intern_opaque_sym(&mut self, hash: F) -> Ptr<F> {
        self.intern_opaque(Tag::Sym, hash)
    }

    pub fn intern_opaque_cons(&mut self, hash: F) -> Ptr<F> {
        self.intern_opaque(Tag::Cons, hash)
    }

    pub fn intern_opaque_comm(&mut self, hash: F) -> Ptr<F> {
        self.intern_opaque(Tag::Comm, hash)
    }

    /// Helper to allocate a list, instead of manually using `cons`.
    pub fn intern_list(&mut self, elts: &[Ptr<F>]) -> Ptr<F> {
        elts.iter()
            .rev()
            .fold(self.sym("nil"), |acc, elt| self.intern_cons(*elt, acc))
    }

    pub(crate) fn convert_sym_case(raw_name: &mut str) {
        // In the future, we could support optional alternate case conventions,
        // so all case conversion should be performed here.
        raw_name.make_ascii_uppercase();
    }

    pub fn intern_sym_with_case_conversion<T: AsRef<str>>(&mut self, name: T) -> Ptr<F> {
        let mut name = name.as_ref().to_string();
        Self::convert_sym_case(&mut name);

        self.intern_sym(name)
    }

    pub fn intern_sym<T: AsRef<str>>(&mut self, name: T) -> Ptr<F> {
        // Hash name for side effect. This will cause all tails to be interned.
        self.hash_string_mut(name.as_ref());
        let name = name.as_ref().to_string();

        let tag = if name == "NIL" { Tag::Nil } else { Tag::Sym };

        if let Some(ptr) = self.sym_store.0.get(&name) {
            Ptr(tag, RawPtr::new(ptr.to_usize()))
        } else {
            let ptr = self.sym_store.0.get_or_intern(name);
            let ptr = Ptr(tag, RawPtr::new(ptr.to_usize()));
            self.dehydrated.push(ptr);
            ptr
        }
    }

    pub fn get_sym<T: AsRef<str>>(&self, name: T, convert_case: bool) -> Option<Ptr<F>> {
        // TODO: avoid allocation
        let mut name = name.as_ref().to_string();
        if convert_case {
            Self::convert_sym_case(&mut name);
        }
        let tag = if name == "NIL" { Tag::Nil } else { Tag::Sym };
        self.sym_store
            .0
            .get(name)
            .map(|raw| Ptr(tag, RawPtr::new(raw.to_usize())))
    }

    pub fn intern_num<T: Into<Num<F>>>(&mut self, num: T) -> Ptr<F> {
        let num = num.into();
        let num = match num {
            Num::Scalar(scalar) => {
                if let Some(u64_num) = scalar.to_u64() {
                    Num::U64(u64_num)
                } else {
                    num
                }
            }
            Num::U64(_) => num,
        };

        let (ptr, _) = self.num_store.insert_full(num);

        Ptr(Tag::Num, RawPtr::new(ptr))
    }

    pub fn get_num<T: Into<Num<F>>>(&self, num: T) -> Option<Ptr<F>> {
        let num = num.into();
        let num = match num {
            Num::Scalar(scalar) => {
                if let Some(u64_num) = scalar.to_u64() {
                    Num::U64(u64_num)
                } else {
                    num
                }
            }
            Num::U64(_) => num,
        };

        self.num_store
            .get_index_of::<Num<F>>(&num)
            .map(|x| Ptr(Tag::Num, RawPtr::new(x)))
    }

    pub fn get_char(&self, c: char) -> Ptr<F> {
        Ptr(Tag::Char, RawPtr::new(u32::from(c) as usize))
    }

    pub fn intern_str<T: AsRef<str>>(&mut self, str: T) -> Ptr<F> {
        // Hash string for side effect. This will cause all tails to be interned.
        self.hash_string_mut(str.as_ref());
        self.intern_str_aux(str)
    }

    fn intern_str_aux<T: AsRef<str>>(&mut self, str: T) -> Ptr<F> {
        if let Some(ptr) = self.str_store.0.get(&str) {
            Ptr(Tag::Str, RawPtr::new(ptr.to_usize()))
        } else {
            let ptr = self.str_store.0.get_or_intern(str);
            let ptr = Ptr(Tag::Str, RawPtr::new(ptr.to_usize()));

            self.dehydrated.push(ptr);
            ptr
        }
    }

    pub fn get_str<T: AsRef<str>>(&self, name: T) -> Option<Ptr<F>> {
        let ptr = self.str_store.0.get(name)?;
        Some(Ptr(Tag::Str, RawPtr::new(ptr.to_usize())))
    }

    pub fn intern_fun(&mut self, arg: Ptr<F>, body: Ptr<F>, closed_env: Ptr<F>) -> Ptr<F> {
        // TODO: closed_env must be an env
        assert!(matches!(arg.0, Tag::Sym), "ARG must be a symbol");
        let (p, inserted) = self.fun_store.insert_full((arg, body, closed_env));
        let ptr = Ptr(Tag::Fun, RawPtr::new(p));
        if inserted {
            self.dehydrated.push(ptr);
        }
        ptr
    }

    pub fn intern_thunk(&mut self, thunk: Thunk<F>) -> Ptr<F> {
        let (p, inserted) = self.thunk_store.insert_full(thunk);
        let ptr = Ptr(Tag::Thunk, RawPtr::new(p));
        if inserted {
            self.dehydrated.push(ptr);
        }
        ptr
    }

    fn mark_dehydrated_cont(&mut self, p: ContPtr<F>) -> ContPtr<F> {
        self.dehydrated_cont.push(p);
        p
    }

    pub fn intern_cont_outermost(&mut self) -> ContPtr<F> {
        self.mark_dehydrated_cont(self.get_cont_outermost())
    }

    pub fn get_cont_outermost(&self) -> ContPtr<F> {
        let ptr = self.sym_store.0.get("OUTERMOST").expect("pre stored");
        ContPtr(ContTag::Outermost, RawPtr::new(ptr.to_usize()))
    }

    pub fn intern_cont_call0(&mut self, a: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.call0_store.insert_full(a);
        let ptr = ContPtr(ContTag::Call0, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_call(&mut self, a: Ptr<F>, b: Ptr<F>, c: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.call_store.insert_full((a, b, c));
        let ptr = ContPtr(ContTag::Call, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_call2(&mut self, a: Ptr<F>, b: Ptr<F>, c: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.call2_store.insert_full((a, b, c));
        let ptr = ContPtr(ContTag::Call2, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_error(&mut self) -> ContPtr<F> {
        self.mark_dehydrated_cont(self.get_cont_error())
    }

    pub fn get_cont_error(&self) -> ContPtr<F> {
        let ptr = self.sym_store.0.get("ERROR").expect("pre stored");
        ContPtr(ContTag::Error, RawPtr::new(ptr.to_usize()))
    }

    pub fn intern_cont_terminal(&mut self) -> ContPtr<F> {
        self.mark_dehydrated_cont(self.get_cont_terminal())
    }

    pub fn get_cont_terminal(&self) -> ContPtr<F> {
        let ptr = self.sym_store.0.get("TERMINAL").expect("pre stored");
        ContPtr(ContTag::Terminal, RawPtr::new(ptr.to_usize()))
    }

    pub fn intern_cont_dummy(&mut self) -> ContPtr<F> {
        self.mark_dehydrated_cont(self.get_cont_dummy())
    }

    pub fn get_cont_dummy(&self) -> ContPtr<F> {
        let ptr = self.sym_store.0.get("DUMMY").expect("pre stored");
        ContPtr(ContTag::Dummy, RawPtr::new(ptr.to_usize()))
    }

    pub fn intern_cont_lookup(&mut self, a: Ptr<F>, b: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.lookup_store.insert_full((a, b));
        let ptr = ContPtr(ContTag::Lookup, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_let(
        &mut self,
        a: Ptr<F>,
        b: Ptr<F>,
        c: Ptr<F>,
        d: ContPtr<F>,
    ) -> ContPtr<F> {
        let (p, inserted) = self.let_store.insert_full((a, b, c, d));
        let ptr = ContPtr(ContTag::Let, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_let_rec(
        &mut self,
        a: Ptr<F>,
        b: Ptr<F>,
        c: Ptr<F>,
        d: ContPtr<F>,
    ) -> ContPtr<F> {
        let (p, inserted) = self.let_rec_store.insert_full((a, b, c, d));
        let ptr = ContPtr(ContTag::LetRec, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_unop(&mut self, op: Op1, a: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.unop_store.insert_full((op, a));
        let ptr = ContPtr(ContTag::Unop, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }
    pub fn intern_cont_emit(&mut self, continuation: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.emit_store.insert_full(continuation);
        let ptr = ContPtr(ContTag::Emit, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_binop(
        &mut self,
        op: Op2,
        a: Ptr<F>,
        b: Ptr<F>,
        c: ContPtr<F>,
    ) -> ContPtr<F> {
        let (p, inserted) = self.binop_store.insert_full((op, a, b, c));
        let ptr = ContPtr(ContTag::Binop, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_binop2(&mut self, op: Op2, a: Ptr<F>, b: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.binop2_store.insert_full((op, a, b));
        let ptr = ContPtr(ContTag::Binop2, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_relop(
        &mut self,
        op: Rel2,
        a: Ptr<F>,
        b: Ptr<F>,
        c: ContPtr<F>,
    ) -> ContPtr<F> {
        let (p, inserted) = self.relop_store.insert_full((op, a, b, c));
        let ptr = ContPtr(ContTag::Relop, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_relop2(&mut self, op: Rel2, a: Ptr<F>, b: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.relop2_store.insert_full((op, a, b));
        let ptr = ContPtr(ContTag::Relop2, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_if(&mut self, a: Ptr<F>, b: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.if_store.insert_full((a, b));
        let ptr = ContPtr(ContTag::If, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn intern_cont_tail(&mut self, a: Ptr<F>, b: ContPtr<F>) -> ContPtr<F> {
        let (p, inserted) = self.tail_store.insert_full((a, b));
        let ptr = ContPtr(ContTag::Tail, RawPtr::new(p));
        if inserted {
            self.dehydrated_cont.push(ptr)
        }
        ptr
    }

    pub fn scalar_from_parts(&self, tag: F, value: F) -> Option<ScalarPtr<F>> {
        let scalar_ptr = ScalarPtr(tag, value);
        if self.scalar_ptr_map.contains_key(&scalar_ptr) {
            return Some(scalar_ptr);
        }

        None
    }

    pub fn scalar_from_parts_cont(&self, tag: F, value: F) -> Option<ScalarContPtr<F>> {
        let scalar_ptr = ScalarContPtr(tag, value);
        if self.scalar_ptr_cont_map.contains_key(&scalar_ptr) {
            return Some(scalar_ptr);
        }
        None
    }

    pub fn fetch_scalar(&self, scalar_ptr: &ScalarPtr<F>) -> Option<Ptr<F>> {
        self.scalar_ptr_map.get(scalar_ptr).map(|p| *p)
    }
    pub fn fetch_scalar_m(&mut self, scalar_ptr: &ScalarPtr<F>) -> Option<Ptr<F>> {
        self.scalar_ptr_map.get(scalar_ptr).map(|p| *p)
    }

    pub fn fetch_scalar_cont(&self, scalar_ptr: &ScalarContPtr<F>) -> Option<ContPtr<F>> {
        self.scalar_ptr_cont_map.get(scalar_ptr).map(|p| *p)
    }

    pub(crate) fn fetch_sym(&self, ptr: &Ptr<F>) -> Option<&str> {
        debug_assert!(matches!(ptr.0, Tag::Sym) | matches!(ptr.0, Tag::Nil));

        if ptr.1.is_opaque() {
            // Ptr.fmt depends on this never returning None for opaque syms.
            if self.opaque_map.contains_key(ptr) {
                return Some("<Opaque Sym>");
            } else {
                // This shouldn't happen.
                return Some("<Opaque Sym [MISSING]>");
            }
        }

        if ptr.0 == Tag::Nil {
            return Some("NIL");
        }

        self.sym_store
            .0
            .resolve(SymbolUsize::try_from_usize(ptr.1.idx()).unwrap())
    }

    pub(crate) fn fetch_str(&self, ptr: &Ptr<F>) -> Option<&str> {
        debug_assert!(matches!(ptr.0, Tag::Str));
        let symbol = SymbolUsize::try_from_usize(ptr.1.idx()).expect("invalid pointer");
        self.str_store.0.resolve(symbol)
    }

    pub(crate) fn fetch_char(&self, ptr: &Ptr<F>) -> Option<char> {
        debug_assert!(matches!(ptr.0, Tag::Char));
        char::from_u32(ptr.1 .0 as u32)
    }

    pub(crate) fn fetch_fun(&self, ptr: &Ptr<F>) -> Option<&(Ptr<F>, Ptr<F>, Ptr<F>)> {
        debug_assert!(matches!(ptr.0, Tag::Fun));
        if ptr.1.is_opaque() {
            None
            // Some(&self.opaque_fun)
        } else {
            self.fun_store.get_index(ptr.1.idx())
        }
    }

    pub(crate) fn fetch_cons(&self, ptr: &Ptr<F>) -> Option<&(Ptr<F>, Ptr<F>)> {
        debug_assert!(matches!(ptr.0, Tag::Cons));
        if ptr.1.is_opaque() {
            None
        } else {
            self.cons_store.get_index(ptr.1.idx())
        }
    }

    pub(crate) fn fetch_comm(&self, ptr: &Ptr<F>) -> Option<&(FWrap<F>, Ptr<F>)> {
        debug_assert!(matches!(ptr.0, Tag::Comm));
        if ptr.1.is_opaque() {
            None
        } else {
            self.comm_store.get_index(ptr.1.idx())
        }
    }

    pub(crate) fn fetch_num(&self, ptr: &Ptr<F>) -> Option<&Num<F>> {
        debug_assert!(matches!(ptr.0, Tag::Num));
        self.num_store.get_index(ptr.1.idx())
    }

    fn fetch_thunk(&self, ptr: &Ptr<F>) -> Option<&Thunk<F>> {
        debug_assert!(matches!(ptr.0, Tag::Thunk));
        self.thunk_store.get_index(ptr.1.idx())
    }

    pub fn fetch(&self, ptr: &Ptr<F>) -> Option<Expression<F>> {
        if ptr.is_opaque() {
            return Some(Expression::Opaque(*ptr));
        }
        match ptr.0 {
            Tag::Nil => Some(Expression::Nil),
            Tag::Cons => self.fetch_cons(ptr).map(|(a, b)| Expression::Cons(*a, *b)),
            Tag::Comm => self.fetch_comm(ptr).map(|(a, b)| Expression::Comm(a.0, *b)),
            Tag::Sym => self.fetch_sym(ptr).map(Expression::Sym),
            Tag::Num => self.fetch_num(ptr).map(|num| Expression::Num(*num)),
            Tag::Fun => self
                .fetch_fun(ptr)
                .map(|(a, b, c)| Expression::Fun(*a, *b, *c)),
            Tag::Thunk => self.fetch_thunk(ptr).map(|thunk| Expression::Thunk(*thunk)),
            Tag::Str => self.fetch_str(ptr).map(|str| Expression::Str(str)),
            Tag::Char => self.fetch_char(ptr).map(Expression::Char),
        }
    }

    pub fn fetch_cont(&self, ptr: &ContPtr<F>) -> Option<Continuation<F>> {
        use ContTag::*;
        match ptr.0 {
            Outermost => Some(Continuation::Outermost),
            Call0 => self
                .call0_store
                .get_index(ptr.1.idx())
                .map(|c| Continuation::Call0 { continuation: *c }),
            Call => self
                .call_store
                .get_index(ptr.1.idx())
                .map(|(a, b, c)| Continuation::Call {
                    unevaled_arg: *a,
                    saved_env: *b,
                    continuation: *c,
                }),
            Call2 => self
                .call2_store
                .get_index(ptr.1.idx())
                .map(|(a, b, c)| Continuation::Call2 {
                    function: *a,
                    saved_env: *b,
                    continuation: *c,
                }),
            Tail => self
                .tail_store
                .get_index(ptr.1.idx())
                .map(|(a, b)| Continuation::Tail {
                    saved_env: *a,
                    continuation: *b,
                }),
            Error => Some(Continuation::Error),
            Lookup => self
                .lookup_store
                .get_index(ptr.1.idx())
                .map(|(a, b)| Continuation::Lookup {
                    saved_env: *a,
                    continuation: *b,
                }),
            Unop => self
                .unop_store
                .get_index(ptr.1.idx())
                .map(|(a, b)| Continuation::Unop {
                    operator: *a,
                    continuation: *b,
                }),
            Binop => {
                self.binop_store
                    .get_index(ptr.1.idx())
                    .map(|(a, b, c, d)| Continuation::Binop {
                        operator: *a,
                        saved_env: *b,
                        unevaled_args: *c,
                        continuation: *d,
                    })
            }
            Binop2 => {
                self.binop2_store
                    .get_index(ptr.1.idx())
                    .map(|(a, b, c)| Continuation::Binop2 {
                        operator: *a,
                        evaled_arg: *b,
                        continuation: *c,
                    })
            }
            Relop => {
                self.relop_store
                    .get_index(ptr.1.idx())
                    .map(|(a, b, c, d)| Continuation::Relop {
                        operator: *a,
                        saved_env: *b,
                        unevaled_args: *c,
                        continuation: *d,
                    })
            }
            Relop2 => {
                self.relop2_store
                    .get_index(ptr.1.idx())
                    .map(|(a, b, c)| Continuation::Relop2 {
                        operator: *a,
                        evaled_arg: *b,
                        continuation: *c,
                    })
            }
            If => self
                .if_store
                .get_index(ptr.1.idx())
                .map(|(a, b)| Continuation::If {
                    unevaled_args: *a,
                    continuation: *b,
                }),
            Let => self
                .let_store
                .get_index(ptr.1.idx())
                .map(|(a, b, c, d)| Continuation::Let {
                    var: *a,
                    body: *b,
                    saved_env: *c,
                    continuation: *d,
                }),
            LetRec => self
                .let_rec_store
                .get_index(ptr.1.idx())
                .map(|(a, b, c, d)| Continuation::LetRec {
                    var: *a,
                    body: *b,
                    saved_env: *c,
                    continuation: *d,
                }),
            Dummy => Some(Continuation::Dummy),
            Terminal => Some(Continuation::Terminal),
            Emit => self
                .emit_store
                .get_index(ptr.1.idx())
                .map(|continuation| Continuation::Emit {
                    continuation: *continuation,
                }),
        }
    }

    /// Mutable version of car_cdr to handle Str. `(cdr str)` may return a new str (the tail), which must be allocated.
    pub fn car_cdr_mut(&mut self, ptr: &Ptr<F>) -> Result<(Ptr<F>, Ptr<F>), &str> {
        match ptr.0 {
            Tag::Nil => Ok((self.get_nil(), self.get_nil())),
            Tag::Cons => match self.fetch(ptr) {
                Some(Expression::Cons(car, cdr)) => Ok((car, cdr)),
                Some(Expression::Opaque(_)) => panic!("cannot destructure opaque Cons"),
                _ => unreachable!(),
            },
            Tag::Str => {
                if let Some(Expression::Str(s)) = self.fetch(ptr) {
                    let mut str = s.chars();
                    if let Some(c) = str.next() {
                        let cdr_str: String = str.collect();
                        let str = self.intern_str(&cdr_str);
                        Ok((self.get_char(c), str))
                    } else {
                        Ok((self.nil(), self.intern_str(&"")))
                    }
                } else {
                    panic!();
                }
            }
            _ => Err("Invalid tag"),
        }
    }

    pub fn car_cdr(&self, ptr: &Ptr<F>) -> (Ptr<F>, Ptr<F>) {
        // FIXME: Maybe make error.
        match ptr.0 {
            Tag::Nil => (self.get_nil(), self.get_nil()),
            Tag::Cons => match self.fetch(ptr) {
                Some(Expression::Cons(car, cdr)) => (car, cdr),
                Some(Expression::Opaque(_)) => panic!("cannot destructure opaque Cons"),
                _ => unreachable!(),
            },
            Tag::Str => {
                if let Some(Expression::Str(s)) = self.fetch(ptr) {
                    let mut chars = s.chars();
                    if let Some(c) = chars.next() {
                        let cdr_str: String = chars.collect();
                        let str = self.get_str(&cdr_str).expect("cdr str missing");
                        (self.get_char(c), str)
                    } else {
                        (self.get_nil(), self.get_str(&"").unwrap())
                    }
                } else {
                    panic!();
                }
            }
            _ => {
                // FIXME: Don't panic. This can happen at runtime in a valid Lurk program,
                // so it should result in an explicit error.
                panic!("Can only extract car_cdr from Cons")
            }
        }
    }

    pub fn open(&mut self, ptr: Ptr<F>) -> Option<Ptr<F>> {
        assert!(ptr.0 == Tag::Comm || ptr.0 == Tag::Num);

        let p = match ptr.0 {
            Tag::Comm => ptr,
            Tag::Num => {
                let scalar = self.fetch_num(&ptr).map(|x| x.into_scalar()).unwrap();

                self.intern_maybe_opaque_comm(scalar)
            }
            _ => return None,
        };

        if let Some((_secret, payload)) = self.fetch_comm(&p) {
            Some(*payload)
        } else {
            None
        }
    }

    pub fn secret(&mut self, ptr: Ptr<F>) -> Option<Ptr<F>> {
        assert_eq!(Tag::Comm, ptr.0);

        let p = match ptr.0 {
            Tag::Comm => ptr,
            Tag::Num => {
                let scalar = self.fetch_num(&ptr).map(|x| x.into_scalar()).unwrap();

                self.intern_maybe_opaque_comm(scalar)
            }
            _ => return None,
        };

        self.fetch_comm(&p)
            .and_then(|(secret, _payload)| self.get_num(Num::Scalar(secret.0)))
    }

    pub fn hash_expr(&self, ptr: &Ptr<F>) -> Option<ScalarPtr<F>> {
        use Tag::*;
        match ptr.tag() {
            Nil => self.hash_nil(),
            Cons => self.hash_cons(*ptr),
            Comm => self.hash_comm(*ptr),
            Sym => self.hash_sym(*ptr),
            Fun => self.hash_fun(*ptr),
            Num => self.hash_num(*ptr),
            Str => self.hash_str(*ptr),
            Char => self.hash_char(*ptr),
            Thunk => self.hash_thunk(*ptr),
        }
    }

    // Get hash for expr, but only if it already exists. This should never cause create_scalar_ptr to be called. Use
    // this after the cache has been hydrated. NOTE: because dashmap::entry can deadlock, it is important not to call
    // hash_expr in nested call graphs which might trigger that behavior. This discovery is what led to get_expr_hash
    // and the 'get' versions of hash_cons, hash_sym, etc.
    pub fn get_expr_hash(&self, ptr: &Ptr<F>) -> Option<ScalarPtr<F>> {
        use Tag::*;
        match ptr.tag() {
            Nil => self.get_hash_nil(),
            Cons => self.get_hash_cons(*ptr),
            Comm => self.get_hash_comm(*ptr),
            Sym => self.get_hash_sym(*ptr),
            Fun => self.get_hash_fun(*ptr),
            Num => self.get_hash_num(*ptr),
            Str => self.get_hash_str(*ptr),
            Char => self.get_hash_char(*ptr),
            Thunk => self.get_hash_thunk(*ptr),
        }
    }

    pub fn hash_cont(&self, ptr: &ContPtr<F>) -> Option<ScalarContPtr<F>> {
        let components = self.get_hash_components_cont(ptr)?;
        let hash = self.poseidon_cache.hash8(&components);

        Some(self.create_cont_scalar_ptr(*ptr, hash))
    }

    /// The only places that `ScalarPtr`s for `Ptr`s should be created, to
    /// ensure that they are cached properly
    fn create_scalar_ptr(&self, ptr: Ptr<F>, hash: F) -> ScalarPtr<F> {
        let scalar_ptr = ScalarPtr(ptr.tag_field(), hash);
        let entry = self.scalar_ptr_map.entry(scalar_ptr);
        entry.or_insert(ptr);
        scalar_ptr
    }

    fn get_scalar_ptr(&self, ptr: Ptr<F>, hash: F) -> ScalarPtr<F> {
        ScalarPtr(ptr.tag_field(), hash)
    }

    /// The only places that `ScalarContPtr`s for `ContPtr`s should be created, to
    /// ensure that they are cached properly
    fn create_cont_scalar_ptr(&self, ptr: ContPtr<F>, hash: F) -> ScalarContPtr<F> {
        let scalar_ptr = ScalarContPtr(ptr.tag_field(), hash);
        self.scalar_ptr_cont_map.entry(scalar_ptr).or_insert(ptr);

        scalar_ptr
    }

    /// The `get_hash_components_*` functions should be kept in sync with the
    /// the arguments of each variant of ScalarContinuation with respect to the
    /// sourc position order of elements
    fn get_hash_components_default(&self) -> [[F; 2]; 4] {
        let def = [F::zero(), F::zero()];
        [def, def, def, def]
    }

    pub fn get_hash_components_cont(&self, ptr: &ContPtr<F>) -> Option<[F; 8]> {
        use Continuation::*;

        let cont = self.fetch_cont(ptr)?;

        let hash = match &cont {
            Outermost | Terminal | Dummy | Error => self.get_hash_components_default(),
            Call0 { continuation } => self.get_hash_components_call0(continuation)?,
            Call {
                unevaled_arg,
                saved_env,
                continuation,
            } => self.get_hash_components_call(unevaled_arg, saved_env, continuation)?,
            Call2 {
                function,
                saved_env,
                continuation,
            } => self.get_hash_components_call2(function, saved_env, continuation)?,
            Tail {
                saved_env,
                continuation,
            } => self.get_hash_components_tail(saved_env, continuation)?,
            Lookup {
                saved_env,
                continuation,
            } => self.get_hash_components_lookup(saved_env, continuation)?,
            Unop {
                operator,
                continuation,
            } => self.get_hash_components_unop(operator, continuation)?,
            Binop {
                operator,
                saved_env,
                unevaled_args,
                continuation,
            } => {
                self.get_hash_components_binop(operator, saved_env, unevaled_args, continuation)?
            }
            Binop2 {
                operator,
                evaled_arg,
                continuation,
            } => self.get_hash_components_binop2(operator, evaled_arg, continuation)?,
            Relop {
                operator,
                saved_env,
                unevaled_args,
                continuation,
            } => {
                self.get_hash_components_relop(operator, saved_env, unevaled_args, continuation)?
            }
            Relop2 {
                operator,
                evaled_arg,
                continuation,
            } => self.get_hash_components_relop2(operator, evaled_arg, continuation)?,
            If {
                unevaled_args,
                continuation,
            } => self.get_hash_components_if(unevaled_args, continuation)?,
            Let {
                var,
                body,
                saved_env,
                continuation,
            } => self.get_hash_components_let(var, body, saved_env, continuation)?,
            LetRec {
                var,
                body,
                saved_env,
                continuation,
            } => self.get_hash_components_let_rec(var, body, saved_env, continuation)?,
            Emit { continuation } => self.get_hash_components_emit(continuation)?,
        };

        Some([
            hash[0][0], hash[0][1], hash[1][0], hash[1][1], hash[2][0], hash[2][1], hash[3][0],
            hash[3][1],
        ])
    }

    fn get_hash_components_let_rec(
        &self,
        var: &Ptr<F>,
        body: &Ptr<F>,
        saved_env: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let var = self.get_expr_hash(var)?.into_hash_components();
        let body = self.get_expr_hash(body)?.into_hash_components();
        let saved_env = self.get_expr_hash(saved_env)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([var, body, saved_env, cont])
    }

    fn get_hash_components_let(
        &self,
        var: &Ptr<F>,
        body: &Ptr<F>,
        saved_env: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let var = self.get_expr_hash(var)?.into_hash_components();
        let body = self.get_expr_hash(body)?.into_hash_components();
        let saved_env = self.get_expr_hash(saved_env)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([var, body, saved_env, cont])
    }

    fn get_hash_components_if(
        &self,
        unevaled_args: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];
        let unevaled_args = self.get_expr_hash(unevaled_args)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([unevaled_args, cont, def, def])
    }

    fn get_hash_components_relop2(
        &self,
        rel: &Rel2,
        arg1: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];
        let rel = self.hash_rel2(rel).into_hash_components();
        let arg1 = self.get_expr_hash(arg1)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([rel, arg1, cont, def])
    }

    fn get_hash_components_relop(
        &self,
        rel: &Rel2,
        saved_env: &Ptr<F>,
        unevaled_args: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let rel = self.hash_rel2(rel).into_hash_components();
        let saved_env = self.get_expr_hash(saved_env)?.into_hash_components();
        let unevaled_args = self.get_expr_hash(unevaled_args)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([rel, saved_env, unevaled_args, cont])
    }

    fn get_hash_components_binop2(
        &self,
        op: &Op2,
        arg1: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];
        let op = self.hash_op2(op).into_hash_components();
        let arg1 = self.get_expr_hash(arg1)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([op, arg1, cont, def])
    }

    fn get_hash_components_binop(
        &self,
        op: &Op2,
        saved_env: &Ptr<F>,
        unevaled_args: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let op = self.hash_op2(op).into_hash_components();
        let saved_env = self.get_expr_hash(saved_env)?.into_hash_components();
        let unevaled_args = self.get_expr_hash(unevaled_args)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([op, saved_env, unevaled_args, cont])
    }

    fn get_hash_components_unop(&self, op: &Op1, cont: &ContPtr<F>) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];
        let op = self.hash_op1(op).into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([op, cont, def, def])
    }

    fn get_hash_components_lookup(
        &self,
        saved_env: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];
        let saved_env = self.get_expr_hash(saved_env)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([saved_env, cont, def, def])
    }

    fn get_hash_components_tail(
        &self,
        saved_env: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];
        let saved_env = self.get_expr_hash(saved_env)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([saved_env, cont, def, def])
    }

    fn get_hash_components_call0(&self, cont: &ContPtr<F>) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];

        let cont = self.hash_cont(cont)?.into_hash_components();

        Some([cont, def, def, def])
    }

    fn get_hash_components_call(
        &self,
        arg: &Ptr<F>,
        saved_env: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];
        let arg = self.get_expr_hash(arg)?.into_hash_components();
        let saved_env = self.get_expr_hash(saved_env)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();

        Some([saved_env, arg, cont, def])
    }

    fn get_hash_components_call2(
        &self,
        fun: &Ptr<F>,
        saved_env: &Ptr<F>,
        cont: &ContPtr<F>,
    ) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];
        let fun = self.get_expr_hash(fun)?.into_hash_components();
        let saved_env = self.get_expr_hash(saved_env)?.into_hash_components();
        let cont = self.hash_cont(cont)?.into_hash_components();
        Some([saved_env, fun, cont, def])
    }

    fn get_hash_components_emit(&self, cont: &ContPtr<F>) -> Option<[[F; 2]; 4]> {
        let def = [F::zero(), F::zero()];

        let cont = self.hash_cont(cont)?.into_hash_components();

        Some([cont, def, def, def])
    }

    pub fn get_hash_components_thunk(&self, thunk: &Thunk<F>) -> Option<[F; 4]> {
        let value_hash = self.get_expr_hash(&thunk.value)?.into_hash_components();
        let continuation_hash = self.hash_cont(&thunk.continuation)?.into_hash_components();

        Some([
            value_hash[0],
            value_hash[1],
            continuation_hash[0],
            continuation_hash[1],
        ])
    }

    pub fn hash_sym(&self, sym: Ptr<F>) -> Option<ScalarPtr<F>> {
        if sym.is_opaque() {
            return self.opaque_map.get(&sym).map(|s| *s);
        }
        let s = self.fetch_sym(&sym)?;

        let str_hash = self.hash_string(s);
        Some(self.create_scalar_ptr(sym, str_hash))
    }

    pub fn get_hash_sym(&self, sym: Ptr<F>) -> Option<ScalarPtr<F>> {
        if sym.is_opaque() {
            return self.opaque_map.get(&sym).map(|s| *s);
        }
        let s = self.fetch_sym(&sym)?;
        Some(self.get_scalar_ptr(sym, self.hash_string(s)))
    }

    fn hash_str(&self, str: Ptr<F>) -> Option<ScalarPtr<F>> {
        if str.is_opaque() {
            return self.opaque_map.get(&str).map(|s| *s);
        }

        let s = self.fetch_str(&str)?;
        Some(self.create_scalar_ptr(str, self.hash_string(s)))
    }

    fn get_hash_str(&self, str: Ptr<F>) -> Option<ScalarPtr<F>> {
        if str.is_opaque() {
            return self.opaque_map.get(&str).map(|s| *s);
        }
        let s = self.fetch_str(&str)?;
        Some(self.get_scalar_ptr(str, self.hash_string(s)))
    }

    fn hash_fun(&self, fun: Ptr<F>) -> Option<ScalarPtr<F>> {
        if fun.is_opaque() {
            Some(
                *self
                    .opaque_map
                    .get(&fun)
                    .expect("ScalarPtr for opaque Fun missing"),
            )
        } else {
            let (arg, body, closed_env) = self.fetch_fun(&fun)?;
            Some(self.create_scalar_ptr(fun, self.hash_ptrs_3(&[*arg, *body, *closed_env])?))
        }
    }
    fn get_hash_fun(&self, fun: Ptr<F>) -> Option<ScalarPtr<F>> {
        if fun.is_opaque() {
            Some(
                *self
                    .opaque_map
                    .get(&fun)
                    .expect("ScalarPtr for opaque Fun missing"),
            )
        } else {
            let (arg, body, closed_env) = self.fetch_fun(&fun)?;
            Some(self.get_scalar_ptr(fun, self.get_hash_ptrs_3(&[*arg, *body, *closed_env])?))
        }
    }

    fn hash_cons(&self, cons: Ptr<F>) -> Option<ScalarPtr<F>> {
        if cons.is_opaque() {
            return Some(
                *self
                    .opaque_map
                    .get(&cons)
                    .expect("ScalarPtr for opaque Cons missing"),
            );
        }

        let (car, cdr) = self.fetch_cons(&cons)?;

        Some(self.create_scalar_ptr(cons, self.hash_ptrs_2(&[*car, *cdr])?))
    }
    fn get_hash_cons(&self, cons: Ptr<F>) -> Option<ScalarPtr<F>> {
        if cons.is_opaque() {
            return Some(
                *self
                    .opaque_map
                    .get(&cons)
                    .expect("ScalarPtr for opaque Cons missing"),
            );
        }

        let (car, cdr) = self.fetch_cons(&cons)?;
        Some(self.get_scalar_ptr(cons, self.get_hash_ptrs_2(&[*car, *cdr])?))
    }

    fn hash_comm(&self, comm: Ptr<F>) -> Option<ScalarPtr<F>> {
        if comm.is_opaque() {
            return Some(
                *self
                    .opaque_map
                    .get(&comm)
                    .expect("ScalarPtr for opaque Comm missing"),
            );
        }

        let (secret, payload) = self.fetch_comm(&comm)?;

        let scalar_payload = self.hash_expr(payload)?;
        let hashed = self.commitment_hash(secret.0, scalar_payload);
        Some(self.create_scalar_ptr(comm, hashed))
    }

    pub(crate) fn commitment_hash(&self, secret_scalar: F, payload: ScalarPtr<F>) -> F {
        let preimage = [
            Tag::Comm.as_field::<F>(),
            secret_scalar,
            payload.0,
            payload.1,
        ];
        self.poseidon_cache.hash4(&preimage)
    }

    fn get_hash_comm(&self, comm: Ptr<F>) -> Option<ScalarPtr<F>> {
        if comm.is_opaque() {
            return Some(
                *self
                    .opaque_map
                    .get(&comm)
                    .expect("ScalarPtr for opaque Comm missing"),
            );
        }

        let (secret, payload) = self.fetch_comm(&comm)?;
        let scalar_payload = self.get_expr_hash(payload)?;
        let hashed = self.commitment_hash(secret.0, scalar_payload);
        Some(self.get_scalar_ptr(comm, hashed))
    }

    fn hash_thunk(&self, ptr: Ptr<F>) -> Option<ScalarPtr<F>> {
        let thunk = self.fetch_thunk(&ptr)?;
        let components = self.get_hash_components_thunk(thunk)?;
        Some(self.create_scalar_ptr(ptr, self.poseidon_cache.hash4(&components)))
    }

    fn get_hash_thunk(&self, ptr: Ptr<F>) -> Option<ScalarPtr<F>> {
        let thunk = self.fetch_thunk(&ptr)?;
        let components = self.get_hash_components_thunk(thunk)?;
        Some(self.get_scalar_ptr(ptr, self.poseidon_cache.hash4(&components)))
    }

    fn hash_char(&self, ptr: Ptr<F>) -> Option<ScalarPtr<F>> {
        let char_code = ptr.1 .0 as u32;

        Some(self.create_scalar_ptr(ptr, F::from(char_code as u64)))
    }
    fn get_hash_char(&self, ptr: Ptr<F>) -> Option<ScalarPtr<F>> {
        let char_code = ptr.1 .0 as u32;

        Some(self.get_scalar_ptr(ptr, F::from(char_code as u64)))
    }

    fn hash_num(&self, ptr: Ptr<F>) -> Option<ScalarPtr<F>> {
        let n = self.fetch_num(&ptr)?;

        Some(self.create_scalar_ptr(ptr, n.into_scalar()))
    }
    fn get_hash_num(&self, ptr: Ptr<F>) -> Option<ScalarPtr<F>> {
        let n = self.fetch_num(&ptr)?;

        Some(self.get_scalar_ptr(ptr, n.into_scalar()))
    }

    fn hash_string(&self, s: &str) -> F {
        if s.is_empty() {
            return F::zero();
        };
        let mut chars = s.chars();
        let char = chars.next().unwrap();
        let rest_string = chars.collect::<String>();
        let c = self.get_char(char);

        let rest = self.get_str(&rest_string).expect("str missing from store");

        self.get_hash_ptrs_2(&[c, rest]).unwrap()
    }

    pub fn hash_string_mut(&mut self, s: &str) -> F {
        let mut v = im::Vector::<char>::new();
        for c in s.chars() {
            v.push_back(c);
        }
        self.intern_str_aux(&s);

        let initial_hash = F::zero();
        let initial_scalar_ptr = {
            let hash = initial_hash;
            let ptr = self.intern_str_aux(&"");
            self.create_scalar_ptr(ptr, hash)
        };

        if s.is_empty() {
            initial_hash
        } else {
            let all_hashes = self.all_hashes(s, initial_scalar_ptr);
            self.hash_string_mut_aux(v, all_hashes)
        }
    }

    // All hashes of substrings, shortest to longest.
    fn all_hashes(&mut self, s: &str, initial_scalar_ptr: ScalarPtr<F>) -> Vec<F> {
        let chars = s.chars().rev();
        let mut hashes = Vec::with_capacity(s.len());

        chars.fold(initial_scalar_ptr, |acc, char| {
            let c_scalar: F = (u32::from(char) as u64).into();
            // This bypasses create_scalar_ptr but is okay because Chars are immediate and don't need to be indexed.
            let c = ScalarPtr(Tag::Char.as_field(), c_scalar);
            let hash = self.hash_scalar_ptrs_2(&[c, acc]);
            // This bypasses create_scalar_ptr but is okay because we will call it to correctly create each of these
            // ScalarPtrs belwo, in hash_string_mut_aux.
            let new_scalar_ptr = ScalarPtr(Tag::Str.as_field(), hash);
            hashes.push(hash);
            new_scalar_ptr
        });

        hashes
    }

    fn hash_string_mut_aux(&mut self, mut s: im::Vector<char>, all_hashes: Vec<F>) -> F {
        debug_assert_eq!(s.len(), all_hashes.len());

        let final_hash = all_hashes.last().unwrap();

        for hash in all_hashes.iter().rev() {
            let string = s.iter().collect::<String>();
            let ptr = self.intern_str_aux(&string);
            self.create_scalar_ptr(ptr, *hash);
            s.slice(..1);
        }

        *final_hash
    }

    fn hash_ptrs_2(&self, ptrs: &[Ptr<F>; 2]) -> Option<F> {
        let scalar_ptrs = [self.hash_expr(&ptrs[0])?, self.hash_expr(&ptrs[1])?];
        Some(self.hash_scalar_ptrs_2(&scalar_ptrs))
    }
    fn get_hash_ptrs_2(&self, ptrs: &[Ptr<F>; 2]) -> Option<F> {
        let scalar_ptrs = [self.get_expr_hash(&ptrs[0])?, self.get_expr_hash(&ptrs[1])?];
        Some(self.hash_scalar_ptrs_2(&scalar_ptrs))
    }

    fn hash_ptrs_3(&self, ptrs: &[Ptr<F>; 3]) -> Option<F> {
        let scalar_ptrs = [
            self.hash_expr(&ptrs[0])?,
            self.hash_expr(&ptrs[1])?,
            self.hash_expr(&ptrs[2])?,
        ];
        Some(self.hash_scalar_ptrs_3(&scalar_ptrs))
    }

    fn get_hash_ptrs_3(&self, ptrs: &[Ptr<F>; 3]) -> Option<F> {
        let scalar_ptrs = [
            self.get_expr_hash(&ptrs[0])?,
            self.get_expr_hash(&ptrs[1])?,
            self.get_expr_hash(&ptrs[2])?,
        ];
        Some(self.hash_scalar_ptrs_3(&scalar_ptrs))
    }

    fn hash_scalar_ptrs_2(&self, ptrs: &[ScalarPtr<F>; 2]) -> F {
        let preimage = [ptrs[0].0, ptrs[0].1, ptrs[1].0, ptrs[1].1];
        self.poseidon_cache.hash4(&preimage)
    }

    fn hash_scalar_ptrs_3(&self, ptrs: &[ScalarPtr<F>; 3]) -> F {
        let preimage = [
            ptrs[0].0, ptrs[0].1, ptrs[1].0, ptrs[1].1, ptrs[2].0, ptrs[2].1,
        ];
        self.poseidon_cache.hash6(&preimage)
    }

    pub fn hash_nil(&self) -> Option<ScalarPtr<F>> {
        let nil = self.get_nil();

        self.hash_sym(nil)
    }

    pub fn get_hash_nil(&self) -> Option<ScalarPtr<F>> {
        let nil = self.get_nil();

        self.get_hash_sym(nil)
    }

    fn hash_op1(&self, op: &Op1) -> ScalarPtr<F> {
        ScalarPtr(op.as_field(), F::zero())
    }

    fn hash_op2(&self, op: &Op2) -> ScalarPtr<F> {
        ScalarPtr(op.as_field(), F::zero())
    }

    fn hash_rel2(&self, op: &Rel2) -> ScalarPtr<F> {
        ScalarPtr(op.as_field(), F::zero())
    }

    // An opaque Ptr is one for which we have the hash, but not the preimages.
    // So we cannot open or traverse the enclosed data, but we can manipulate
    // it atomically and include it in containing structures, etc.
    pub fn new_opaque_ptr(&mut self) -> Ptr<F> {
        // TODO: May need new tag for this.
        // Meanwhile, it is illegal to try to dereference/follow an opaque PTR.
        // So any tag and RawPtr are okay.
        Ptr(Tag::Nil, self.new_opaque_raw_ptr())
    }

    pub fn new_opaque_raw_ptr(&mut self) -> RawPtr<F> {
        self.opaque_raw_ptr_count += 1;
        let p = self.opaque_raw_ptr_count;
        // We can't use 0 for both opaque and normal RawPtrs.
        assert!(p > 0);
        // Ensure there will be no overlow.
        assert!(p < isize::MAX as usize);
        RawPtr(0isize - p as isize, Default::default())
    }

    pub fn ptr_eq(&self, a: &Ptr<F>, b: &Ptr<F>) -> bool {
        // In order to compare Ptrs, we *must* resolve the hashes. Otherwise, we risk failing to recognize equality of
        // compound data with opaque data in either element's transitive closure.
        self.get_expr_hash(a) == self.get_expr_hash(b)
    }

    pub fn cons_eq(&self, a: &Ptr<F>, b: &Ptr<F>) -> bool {
        assert_eq!(Tag::Cons, a.tag());
        assert_eq!(Tag::Cons, b.tag());

        let a_opaque = a.is_opaque();
        let b_opaque = b.is_opaque();

        if !a_opaque && !b_opaque {
            return a == b;
        }
        self.get_expr_hash(a) == self.get_expr_hash(b)
    }

    /// Fill the cache for Scalars. Only Ptrs which have been interned since last hydration will be hashed, so it is
    /// safe to call this incrementally. However, for best proving performance, we should call exactly once so all
    /// hashing can be batched, e.g. on the GPU.
    pub fn hydrate_scalar_cache(&mut self) {
        self.dehydrated.par_iter().for_each(|ptr| {
            self.hash_expr(ptr).expect("failed to hash_expr");
        });

        self.dehydrated.truncate(0);

        self.dehydrated_cont.par_iter().for_each(|ptr| {
            self.hash_cont(ptr).expect("failed to hash_cont");
        });

        self.dehydrated_cont.truncate(0);

        self.dehydrated_cont.clear();
    }
}

impl<F: LurkField> Expression<'_, F> {
    pub fn is_keyword_sym(&self) -> bool {
        match self {
            Expression::Sym(s) => s.starts_with(':'),
            _ => false,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Expression::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_sym_str(&self) -> Option<&str> {
        match self {
            Expression::Sym(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Nil)
    }

    pub fn is_cons(&self) -> bool {
        matches!(self, Self::Cons(_, _))
    }

    pub fn is_list(&self) -> bool {
        self.is_null() || self.is_cons()
    }

    pub fn is_sym(&self) -> bool {
        matches!(self, Self::Sym(_))
    }
    pub fn is_fun(&self) -> bool {
        matches!(self, Self::Fun(_, _, _))
    }

    pub fn is_num(&self) -> bool {
        matches!(self, Self::Num(_))
    }
    pub fn is_str(&self) -> bool {
        matches!(self, Self::Str(_))
    }

    pub fn is_thunk(&self) -> bool {
        matches!(self, Self::Thunk(_))
    }

    pub fn is_opaque(&self) -> bool {
        matches!(self, Self::Opaque(_))
    }
}

#[cfg(test)]
pub mod test {
    use crate::eval::{empty_sym_env, Evaluator};
    use crate::field::FWrap;
    use crate::num;
    use crate::writer::Write;
    use blstrs::Scalar as Fr;

    use super::*;
    use quickcheck::{Arbitrary, Gen};

    use crate::test::frequency;

    use libipld::serde::from_ipld;
    use libipld::serde::to_ipld;
    use libipld::Ipld;

    impl Arbitrary for Tag {
        fn arbitrary(g: &mut Gen) -> Self {
            let input: Vec<(i64, Box<dyn Fn(&mut Gen) -> Tag>)> = vec![
                (100, Box::new(|_| Tag::Nil)),
                (100, Box::new(|_| Tag::Cons)),
                (100, Box::new(|_| Tag::Sym)),
                (100, Box::new(|_| Tag::Fun)),
                (100, Box::new(|_| Tag::Num)),
                (100, Box::new(|_| Tag::Thunk)),
                (100, Box::new(|_| Tag::Str)),
            ];
            frequency(g, input)
        }
    }
    impl Arbitrary for ContTag {
        fn arbitrary(g: &mut Gen) -> Self {
            let input: Vec<(i64, Box<dyn Fn(&mut Gen) -> ContTag>)> = vec![
                (100, Box::new(|_| ContTag::Outermost)),
                (100, Box::new(|_| ContTag::Call)),
                (100, Box::new(|_| ContTag::Call2)),
                (100, Box::new(|_| ContTag::Tail)),
                (100, Box::new(|_| ContTag::Error)),
                (100, Box::new(|_| ContTag::Lookup)),
                (100, Box::new(|_| ContTag::Unop)),
                (100, Box::new(|_| ContTag::Binop)),
                (100, Box::new(|_| ContTag::Binop2)),
                (100, Box::new(|_| ContTag::Relop)),
                (100, Box::new(|_| ContTag::Relop2)),
                (100, Box::new(|_| ContTag::If)),
                (100, Box::new(|_| ContTag::Let)),
                (100, Box::new(|_| ContTag::LetRec)),
                (100, Box::new(|_| ContTag::Dummy)),
                (100, Box::new(|_| ContTag::Terminal)),
                (100, Box::new(|_| ContTag::Emit)),
            ];
            frequency(g, input)
        }
    }

    impl Arbitrary for ScalarPtr<Fr> {
        fn arbitrary(g: &mut Gen) -> Self {
            let tag = Tag::arbitrary(g);
            let val = FWrap::arbitrary(g);
            ScalarPtr::from_parts(Fr::from(tag as u64), val.0)
        }
    }

    #[quickcheck]
    fn test_scalar_ptr_ipld(x: ScalarPtr<Fr>) -> bool {
        if let Ok(ipld) = to_ipld(x) {
            if let Ok(y) = from_ipld(ipld) {
                x == y
            } else {
                false
            }
        } else {
            false
        }
    }

    #[test]
    fn unit_scalar_ptr_ipld() {
        let tag = Tag::Num.as_field();
        let dig = 0.into();
        let ptr = ScalarPtr::<Fr>::from_parts(tag, dig);
        let cid = Cid::new_v1(Fr::to_multicodec(tag).unwrap(), Fr::to_multihash(dig));
        assert_eq!(to_ipld(ptr).unwrap(), Ipld::Link(cid))
    }

    impl Arbitrary for ScalarContPtr<Fr> {
        fn arbitrary(g: &mut Gen) -> Self {
            let tag = ContTag::arbitrary(g);
            let val = FWrap::arbitrary(g);
            ScalarContPtr::from_parts(Fr::from(tag as u64), val.0)
        }
    }

    #[quickcheck]
    fn prop_scalar_cont_ptr_ipld(x: ScalarContPtr<Fr>) -> bool {
        if let Ok(ipld) = to_ipld(x) {
            if let Ok(y) = from_ipld(ipld) {
                x == y
            } else {
                false
            }
        } else {
            false
        }
    }

    #[test]
    fn unit_scalar_cont_ptr_ipld() {
        let tag = ContTag::Dummy.as_field();
        let dig = 0.into();
        let ptr = ScalarContPtr::<Fr>::from_parts(tag, dig);
        let cid = Cid::new_v1(Fr::to_multicodec(tag).unwrap(), Fr::to_multihash(dig));
        assert_eq!(to_ipld(ptr).unwrap(), Ipld::Link(cid))
    }

    impl Arbitrary for Op1 {
        fn arbitrary(g: &mut Gen) -> Self {
            let input: Vec<(i64, Box<dyn Fn(&mut Gen) -> Op1>)> = vec![
                (100, Box::new(|_| Op1::Car)),
                (100, Box::new(|_| Op1::Cdr)),
                (100, Box::new(|_| Op1::Atom)),
                (100, Box::new(|_| Op1::Emit)),
                (100, Box::new(|_| Op1::Secret)),
                (100, Box::new(|_| Op1::Commit)),
                (100, Box::new(|_| Op1::Num)),
                (100, Box::new(|_| Op1::Comm)),
            ];
            frequency(g, input)
        }
    }

    #[quickcheck]
    fn prop_op1_ipld(x: Op1) -> bool {
        if let Ok(ipld) = to_ipld(x) {
            if let Ok(y) = from_ipld(ipld) {
                x == y
            } else {
                false
            }
        } else {
            false
        }
    }

    #[test]
    fn unit_op1_ipld() {
        assert_eq!(
            to_ipld(Op1::Car).unwrap(),
            Ipld::Integer(0b0010_0000_0000_0000 as i128)
        );
    }

    impl Arbitrary for Op2 {
        fn arbitrary(g: &mut Gen) -> Self {
            let input: Vec<(i64, Box<dyn Fn(&mut Gen) -> Op2>)> = vec![
                (100, Box::new(|_| Op2::Sum)),
                (100, Box::new(|_| Op2::Diff)),
                (100, Box::new(|_| Op2::Product)),
                (100, Box::new(|_| Op2::Quotient)),
                (100, Box::new(|_| Op2::Cons)),
                (100, Box::new(|_| Op2::Hide)),
            ];
            frequency(g, input)
        }
    }

    #[quickcheck]
    fn prop_op2_ipld_embed(x: Op2) -> bool {
        if let Ok(ipld) = to_ipld(x) {
            if let Ok(y) = from_ipld(ipld) {
                x == y
            } else {
                false
            }
        } else {
            false
        }
    }

    #[test]
    fn unit_op2_ipld() {
        assert_eq!(
            to_ipld(Op2::Sum).unwrap(),
            Ipld::Integer(0b0011_0000_0000_0000 as i128)
        );
    }

    impl Arbitrary for Rel2 {
        fn arbitrary(g: &mut Gen) -> Self {
            let input: Vec<(i64, Box<dyn Fn(&mut Gen) -> Rel2>)> = vec![
                (100, Box::new(|_| Rel2::Equal)),
                (100, Box::new(|_| Rel2::NumEqual)),
            ];
            frequency(g, input)
        }
    }

    #[quickcheck]
    fn prop_rel2_ipld(x: Rel2) -> bool {
        if let Ok(ipld) = to_ipld(x) {
            if let Ok(y) = from_ipld(ipld) {
                x == y
            } else {
                false
            }
        } else {
            false
        }
    }

    #[test]
    fn unit_rel2_ipld() {
        assert_eq!(
            to_ipld(Rel2::Equal).unwrap(),
            Ipld::Integer(0b0100_0000_0000_0000 as i128)
        );
    }

    #[test]
    fn test_print_num() {
        let mut store = Store::<Fr>::default();
        let num = store.num(5);
        let res = num.fmt_to_string(&store);
        assert_eq!(&res, &"5");
    }

    #[test]
    fn tag_vals() {
        assert_eq!(0, Tag::Nil as u64);
        assert_eq!(1, Tag::Cons as u64);
        assert_eq!(2, Tag::Sym as u64);
        assert_eq!(3, Tag::Fun as u64);
        assert_eq!(4, Tag::Num as u64);
        assert_eq!(5, Tag::Thunk as u64);
        assert_eq!(6, Tag::Str as u64);
        assert_eq!(7, Tag::Char as u64);
        assert_eq!(8, Tag::Comm as u64);
    }

    #[test]
    fn cont_tag_vals() {
        use super::ContTag::*;

        assert_eq!(0b0001_0000_0000_0000, Outermost as u16);
        assert_eq!(0b0001_0000_0000_0001, Call0 as u16);
        assert_eq!(0b0001_0000_0000_0010, Call as u16);
        assert_eq!(0b0001_0000_0000_0011, Call2 as u16);
        assert_eq!(0b0001_0000_0000_0100, Tail as u16);
        assert_eq!(0b0001_0000_0000_0101, Error as u16);
        assert_eq!(0b0001_0000_0000_0110, Lookup as u16);
        assert_eq!(0b0001_0000_0000_0111, Unop as u16);
        assert_eq!(0b0001_0000_0000_1000, Binop as u16);
        assert_eq!(0b0001_0000_0000_1001, Binop2 as u16);
        assert_eq!(0b0001_0000_0000_1010, Relop as u16);
        assert_eq!(0b0001_0000_0000_1011, Relop2 as u16);
        assert_eq!(0b0001_0000_0000_1100, If as u16);
        assert_eq!(0b0001_0000_0000_1101, Let as u16);
        assert_eq!(0b0001_0000_0000_1110, LetRec as u16);
        assert_eq!(0b0001_0000_0000_1111, Dummy as u16);
        assert_eq!(0b0001_0000_0001_0000, Terminal as u16);
    }

    #[test]
    fn store() {
        let mut store = Store::<Fr>::default();

        let num_ptr = store.num(123);
        let num = store.fetch(&num_ptr).unwrap();
        let num_again = store.fetch(&num_ptr).unwrap();

        assert_eq!(num, num_again);
    }

    #[test]
    fn equality() {
        let mut store = Store::<Fr>::default();

        let (a, b) = (store.num(123), store.sym("pumpkin"));
        let cons1 = store.intern_cons(a, b);
        let (a, b) = (store.num(123), store.sym("pumpkin"));
        let cons2 = store.intern_cons(a, b);

        assert_eq!(cons1, cons2);
        assert_eq!(store.car(&cons1), store.car(&cons2));
        assert_eq!(store.cdr(&cons1), store.cdr(&cons2));

        let (a, d) = store.car_cdr(&cons1);
        assert_eq!(store.car(&cons1), a);
        assert_eq!(store.cdr(&cons1), d);
    }

    #[test]
    fn opaque_fun() {
        let mut store = Store::<Fr>::default();

        let arg = store.sym("A");
        let body_form = store.num(123);
        let body2_form = store.num(987);
        let body = store.list(&[body_form]);
        let body2 = store.list(&[body2_form]);
        let empty_env = empty_sym_env(&store);
        let fun = store.intern_fun(arg, body, empty_env);
        let fun2 = store.intern_fun(arg, body2, empty_env);
        let fun_hash = store.hash_expr(&fun).unwrap();
        let fun_hash2 = store.hash_expr(&fun2).unwrap();
        let opaque_fun = store.intern_opaque_fun(*fun_hash.value());
        let opaque_fun2 = store.intern_opaque_fun(*fun_hash2.value());

        let eq = store.sym("eq");
        let t = store.sym("t");
        let nil = store.nil();
        let limit = 10;
        {
            let comparison_expr = store.list(&[eq, fun, opaque_fun]);
            println!("comparison_expr: {}", comparison_expr.fmt_to_string(&store));
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(t, result.expr);
        }
        {
            let comparison_expr = store.list(&[eq, fun2, opaque_fun]);
            println!("comparison_expr: {}", comparison_expr.fmt_to_string(&store));
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(nil, result.expr);
        }
        {
            let comparison_expr = store.list(&[eq, fun2, opaque_fun2]);
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(t, result.expr);
        }
        {
            // This test is important. It demonstrates that we can handle opaque data in compound data being evaluated
            // without this affecting equality semantics.

            let n = store.num(123);
            let cons = store.sym("cons");
            let cons_expr1 = store.list(&[cons, fun, n]);
            let cons_expr2 = store.list(&[cons, opaque_fun, n]);

            let comparison_expr = store.list(&[eq, cons_expr1, cons_expr2]);
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(t, result.expr);
        }
    }

    #[test]
    fn opaque_sym() {
        let mut store = Store::<Fr>::default();

        let empty_env = empty_sym_env(&store);
        let sym = store.sym(&"sym");
        let sym2 = store.sym(&"sym2");
        let sym_hash = store.hash_expr(&sym).unwrap();
        let sym_hash2 = store.hash_expr(&sym2).unwrap();
        let opaque_sym = store.intern_opaque_sym(*sym_hash.value());
        let opaque_sym2 = store.intern_opaque_sym(*sym_hash2.value());

        let quote = store.sym("quote");
        let qsym = store.list(&[quote, sym]);
        let qsym2 = store.list(&[quote, sym2]);
        let qsym_opaque = store.list(&[quote, opaque_sym]);
        let qsym_opaque2 = store.list(&[quote, opaque_sym2]);

        let eq = store.sym("eq");
        let t = store.sym("t");
        let nil = store.nil();
        let limit = 10;

        // When an opaque sym is inserted into a store which contains the same sym, the store knows its identity.
        // Should we just immediately coalesce and never create an opaque version in that case? Probably not because
        // that may interact badly with explicit hiding to be implemented.
        // Let's defer any such considerations until we have a well-specified way of segreting secret/private data.
        //
        // If implemented, the following commented test would pass.
        // assert_eq!(sym.fmt_to_string(&store), opaque_sym.fmt_to_string(&store));

        // For now, all opaque data remains opaque, even if the Store has enough information to clarify it.
        assert!(sym.fmt_to_string(&store) != opaque_sym.fmt_to_string(&store));

        let mut other_store = Store::<Fr>::default();
        let other_opaque_sym = other_store.intern_opaque_sym(*sym_hash.value());

        let other_sym = other_store.sym("sym");
        // other_sym and other_opaque_sym are not equal, since the non-opaque symbol was inserted after the opaque one.
        // TODO: we could check for this and fix when inserting non-opaque syms. If we decide to clarify opaque data
        // when possible, we should do this too.
        assert!(
            other_sym.fmt_to_string(&other_store) != other_opaque_sym.fmt_to_string(&other_store)
        );

        let num = num::Num::from_scalar(*sym_hash.value());
        assert_eq!(
            format!(
                "<Opaque Sym {}>",
                Expression::Num(num).fmt_to_string(&store)
            ),
            other_opaque_sym.fmt_to_string(&other_store)
        );

        // We need to insert a few opaque syms in other_store, in order to acquire a raw_ptr that doesn't exist in
        // store. Use that to check for a malformed/missing opaque sym in store below.
        let _other_opaque_sym2 = other_store.intern_opaque_sym(*sym_hash.value());
        let other_opaque_sym3 = other_store.intern_opaque_sym(*sym_hash.value());

        // other_opaque_sym doesn't exist at all in store, but it is recognized as an opaque sym.
        // It still prints 'normally', but attempts to fetch its name detect this case.
        // This shouldn't actually happen. The test just exercise the code path which detects it.
        assert_eq!("<Opaque Sym>", other_opaque_sym3.fmt_to_string(&store));
        assert_eq!(
            "<Opaque Sym [MISSING]>",
            store.fetch_sym(&other_opaque_sym3).unwrap()
        );

        {
            let comparison_expr = store.list(&[eq, qsym, qsym_opaque]);
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(t, result.expr);
        }
        {
            let comparison_expr = store.list(&[eq, qsym2, qsym_opaque]);
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(nil, result.expr);
        }
        {
            let comparison_expr = store.list(&[eq, qsym2, qsym_opaque2]);
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(t, result.expr);
        }
        {
            // This test is important. It demonstrates that we can handle opaque data in compound data being evaluated
            // without this affecting equality semantics.

            let n = store.num(123);
            let cons = store.sym("cons");
            let cons_expr1 = store.list(&[cons, qsym, n]);
            let cons_expr2 = store.list(&[cons, qsym_opaque, n]);

            let comparison_expr = store.list(&[eq, cons_expr1, cons_expr2]);
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(t, result.expr);
        }
    }

    #[test]
    fn opaque_cons() {
        let mut store = Store::<Fr>::default();

        let num1 = store.num(123);
        let num2 = store.num(987);
        let empty_env = empty_sym_env(&store);
        let cons = store.intern_cons(num1, num2);
        let cons2 = store.intern_cons(num2, num1);
        let cons_hash = store.hash_expr(&cons).unwrap();
        let cons_hash2 = store.hash_expr(&cons2).unwrap();
        let opaque_cons = store.intern_opaque_cons(*cons_hash.value());
        let opaque_cons2 = store.intern_opaque_cons(*cons_hash2.value());

        let eq = store.sym("eq");
        let t = store.sym("t");
        let nil = store.nil();
        let limit = 10;
        let quote = store.sym("quote");
        let qcons = store.list(&[quote, cons]);
        let qcons2 = store.list(&[quote, cons2]);
        let qcons_opaque = store.list(&[quote, opaque_cons]);
        let qcons_opaque2 = store.list(&[quote, opaque_cons2]);

        let num = Expression::Num(num::Num::Scalar(*cons_hash.value()));
        assert_eq!(
            format!("<Opaque Cons {}>", num.fmt_to_string(&store)),
            opaque_cons.fmt_to_string(&store)
        );

        {
            let comparison_expr = store.list(&[eq, qcons, qcons_opaque]);
            // FIXME: need to implement Write for opaque data.
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(t, result.expr);
        }
        {
            let comparison_expr = store.list(&[eq, qcons2, qcons_opaque]);
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(nil, result.expr);
        }
        {
            let comparison_expr = store.list(&[eq, qcons2, qcons_opaque2]);
            let (result, _, _) =
                Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
            assert_eq!(t, result.expr);
        }
        {
            // This test is important. It demonstrates that we can handle opaque data in compound data being evaluated
            // without this affecting equality semantics.

            let n = store.num(123);
            let n2 = store.num(321);
            let cons_sym = store.sym("cons");
            let cons_expr1 = store.list(&[cons_sym, qcons, n]);
            let cons_expr2 = store.list(&[cons_sym, qcons_opaque, n]);
            let cons_expr3 = store.list(&[cons_sym, qcons_opaque, n2]);

            let comparison_expr = store.list(&[eq, cons_expr1, cons_expr2]);
            let comparison_expr2 = store.list(&[eq, cons_expr1, cons_expr3]);
            {
                let (result, _, _) =
                    Evaluator::new(comparison_expr, empty_env, &mut store, limit).eval();
                assert_eq!(t, result.expr);
            }
            {
                let (result, _, _) =
                    Evaluator::new(comparison_expr2, empty_env, &mut store, limit).eval();
                assert_eq!(nil, result.expr);
            }
        }
    }

    fn make_maybe_opaque_cons(store: &mut Store<Fr>, car: u64, cdr: u64) -> Ptr<Fr> {
        let num1 = store.num(Num::<Fr>::from(car));
        let num2 = store.num(Num::<Fr>::from(cdr));
        let cons = store.intern_cons(num1, num2);
        let cons_hash = store.hash_expr(&cons).unwrap();

        store.intern_maybe_opaque_cons(*cons_hash.value())
    }

    fn make_opaque_cons(store: &mut Store<Fr>) -> Ptr<Fr> {
        store.intern_opaque_cons(12345.into())
    }

    #[test]
    #[should_panic]
    fn opaque_cons_car() {
        let mut store = Store::<Fr>::default();

        let opaque_cons = make_opaque_cons(&mut store);
        store.car(&opaque_cons);
    }
    #[test]
    #[should_panic]
    fn opaque_cons_cdr() {
        let mut store = Store::<Fr>::default();

        let opaque_cons = make_opaque_cons(&mut store);
        store.cdr(&opaque_cons);
    }

    #[test]
    fn maybe_opaque_cons_car() {
        let mut store = Store::<Fr>::default();

        let opaque_cons = make_maybe_opaque_cons(&mut store, 123, 987);
        store.car(&opaque_cons);
    }
    #[test]
    fn maybe_opaque_cons_cdr() {
        let mut store = Store::<Fr>::default();

        let opaque_cons = make_maybe_opaque_cons(&mut store, 123, 987);
        store.cdr(&opaque_cons);
    }

    #[test]
    fn str_and_sym_hashes() {
        let s = &mut Store::<Fr>::default();

        let sym = s.sym("orange");
        let str = s.read(r#" "ORANGE" "#).unwrap();

        let sym_hash = s.get_expr_hash(&sym).unwrap().1;
        let str_hash = s.hash_expr(&str).unwrap().1;

        assert_eq!(sym_hash, str_hash);
    }

    #[test]
    fn empty_str_tag_hash() {
        let s = &mut Store::<Fr>::default();

        let sym = s.sym("");
        let sym_tag = s.get_expr_hash(&sym).unwrap().0;
        let sym_hash = s.get_expr_hash(&sym).unwrap().1;

        assert_eq!(Tag::Sym.as_field::<Fr>(), sym_tag);
        assert_eq!(Fr::from(0), sym_hash)
    }

    #[test]
    fn str_car_cdr_hashes() {
        let s = &mut Store::<Fr>::default();

        let str = s.read(r#" "ORANGE" "#).unwrap();
        let str2 = s.cdr(&str);
        let c = s.car(&str);

        let str_hash = s.hash_expr(&str).unwrap().1;

        let str_again = s.cons(c, str2);
        let str_again_hash = s.hash_expr(&str_again).unwrap().1;

        assert_eq!(str_hash, str_again_hash);
    }

    fn str_inner_fetch_aux(str: &str, hydrate: bool) {
        let s = &mut Store::<Fr>::default();

        let str = s.read(str).unwrap();
        let str2 = s.cdr(&str);

        // Unless the cache is hydrated, the inner destructuring will not map the ScalarPtr to corresponding Ptr.
        if hydrate {
            s.hydrate_scalar_cache();
        };

        let str2_scalar_ptr = s.get_expr_hash(&str2).unwrap();

        let str2_again = s.fetch_scalar(&str2_scalar_ptr).unwrap();

        assert_eq!(str2, str2_again);
    }

    #[test]
    fn str_inner_fetch_hydrated() {
        str_inner_fetch_aux(r#" "ORANGE" "#, true);
    }

    #[test]
    fn str_inner_fetch_unhydrated() {
        str_inner_fetch_aux(r#" "ORANGE" "#, false);
    }

    fn empty_str_fetch_aux(hydrate: bool) {
        let s = &mut Store::<Fr>::default();

        let str = s.read(r#" "" "#).unwrap();

        // Unless the cache is hydrated, the inner destructuring will not map the ScalarPtr to corresponding Ptr.
        if hydrate {
            s.hydrate_scalar_cache();
        };

        let str_scalar_ptr = s.get_expr_hash(&str).unwrap();

        let str_again = s.fetch_scalar(&str_scalar_ptr).unwrap();

        assert_eq!(str, str_again);
    }
    #[test]
    fn empty_str_fetch_hydrated() {
        empty_str_fetch_aux(true);
    }

    #[test]
    fn empty_str_fetch_unhydrated() {
        empty_str_fetch_aux(false);
    }

    #[test]
    fn opaque_comm_fmt() {
        let s = &mut Store::<Fr>::default();

        let scalar = Fr::from(123);
        let opaque_comm = s.intern_opaque_comm(Fr::from(123));
        let expr = s.fetch(&opaque_comm).unwrap();

        let num = num::Num::from_scalar(scalar);
        assert_eq!(
            format!("<Opaque Comm {}>", Expression::Num(num).fmt_to_string(s)),
            opaque_comm.fmt_to_string(&s),
        );

        assert_eq!(opaque_comm.fmt_to_string(&s), expr.fmt_to_string(&s));
    }
}
