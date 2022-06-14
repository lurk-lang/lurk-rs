use log::info;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;
use std::str::FromStr;

use bellperson::{groth16, SynthesisError};
use blstrs::{Bls12, Scalar};
use ff::PrimeField;
use hex::FromHex;
use libipld::{
    cbor::DagCborCodec,
    json::DagJsonCodec,
    multihash::{Code, MultihashDigest},
    prelude::Codec,
    serde::{from_ipld, to_ipld},
    Cid, Ipld,
};
use lurk::{
    circuit::ToInputs,
    eval::{empty_sym_env, Evaluable, Evaluator, Status, IO},
    field::LurkField,
    proof::{
        self,
        groth16::{Groth16, Groth16Prover, INNER_PRODUCT_SRS},
    },
    scalar_store::ScalarStore,
    store::{Pointer, Ptr, ScalarPointer, ScalarPtr, Store, Tag},
    writer::Write,
    Num,
};
use once_cell::sync::OnceCell;
use pairing_lib::{Engine, MultiMillerLoop};
use rand::rngs::OsRng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

mod file_map;

use file_map::FileMap;

pub const DEFAULT_REDUCTION_COUNT: ReductionCount = ReductionCount::One;
pub static VERBOSE: OnceCell<bool> = OnceCell::new();

mod base64 {
    use serde::{Deserialize, Serialize};
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        let base64 = base64::encode(v);
        String::serialize(&base64, s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let base64 = String::deserialize(d)?;
        base64::decode(base64.as_bytes()).map_err(serde::de::Error::custom)
    }
}

pub mod ipfs;

fn bls12_proof_cache() -> FileMap<Cid, Proof<Bls12>> {
    FileMap::<Cid, Proof<Bls12>>::new("bls12_proofs").unwrap()
}

pub fn committed_function_store() -> FileMap<Commitment<Scalar>, Function<Scalar>> {
    FileMap::<Commitment<Scalar>, Function<Scalar>>::new("functions").unwrap()
}

fn get_pvk(rc: ReductionCount) -> groth16::PreparedVerifyingKey<Bls12> {
    let groth_prover = Groth16Prover::new(rc.reduction_frame_count());

    info!("Getting Parameters");
    let groth_params = groth_prover.groth_params().unwrap();

    info!("Preparing verifying key");
    groth16::prepare_verifying_key(&groth_params.vk)
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionCount {
    One,
    Five,
    Ten,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct Evaluation {
    pub expr: String,
    pub env: String,
    pub cont: String,
    pub expr_out: String,
    pub env_out: String,
    pub cont_out: String,
    pub status: Status,
    pub iterations: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Commitment<F: LurkField> {
    pub comm: F,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpeningRequest<F: LurkField> {
    pub commitment: Commitment<F>,
    pub input: Expression,
    pub chain: bool,
}

impl<F: LurkField> ToString for Commitment<F> {
    fn to_string(&self) -> String {
        let s = serde_json::to_string(&self).unwrap();
        // Remove quotation marks. Yes, dumb hacks are happening.
        s[1..s.len() - 1].to_string()
    }
}

impl<F: LurkField> Serialize for Commitment<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Use be_bytes for consistency with PrimeField printed representation.
        let be_bytes: Vec<u8> = self
            .comm
            .to_repr()
            .as_ref()
            .iter()
            .rev()
            .map(|x| x.to_owned())
            .collect();

        hex::serde::serialize(be_bytes, serializer)
    }
}

impl<'de, F: LurkField> Deserialize<'de> for Commitment<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        hex::serde::deserialize(deserializer)
    }
}

impl<F: LurkField> FromHex for Commitment<F> {
    type Error = hex::FromHexError;

    fn from_hex<T>(s: T) -> Result<Self, <Self as FromHex>::Error>
    where
        T: AsRef<[u8]>,
    {
        let mut v = Vec::from_hex(s)?;
        v.reverse();
        let mut repr = <F as PrimeField>::Repr::default();
        repr.as_mut()[..32].copy_from_slice(&v[..]);

        Ok(Commitment {
            comm: F::from_repr(repr).unwrap(),
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Expression {
    pub expr: LurkPtr,
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct Opening<F: LurkField> {
    pub input: String,
    pub output: String,
    pub status: Status,
    pub commitment: Commitment<F>,
    pub new_commitment: Option<Commitment<F>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct LurkScalarBytes {
    #[serde(with = "base64")]
    scalar_store: Vec<u8>,
    #[serde(with = "base64")]
    scalar_ptr: Vec<u8>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LurkScalarIpld {
    scalar_store: Ipld,
    scalar_ptr: Ipld,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum LurkPtr {
    Source(String),
    ScalarBytes(LurkScalarBytes),
    Ipld(LurkScalarIpld),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Function<F: LurkField + Serialize> {
    pub fun: LurkPtr,
    #[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
    pub secret: Option<F>,
    pub commitment: Option<Commitment<F>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VerificationResult {
    pub verified: bool,
}

#[derive(Serialize, Deserialize)]
pub struct Proof<E: Engine + MultiMillerLoop>
where
    <E as Engine>::Gt: blstrs::Compress + Serialize,
    <E as Engine>::G1: Serialize,
    <E as Engine>::G1Affine: Serialize,
    <E as Engine>::G2Affine: Serialize,
    <E as Engine>::Fr: Serialize + LurkField,
    <E as Engine>::Gt: blstrs::Compress + Serialize,
{
    #[serde(bound(
        serialize = "Claim<E::Fr>: Serialize",
        deserialize = "Claim<E::Fr>: Deserialize<'de>"
    ))]
    pub claim: Claim<E::Fr>,
    #[serde(bound(
        serialize = "proof::groth16::Proof<E>: Serialize",
        deserialize = "proof::groth16::Proof<E>: Deserialize<'de>"
    ))]
    pub proof: proof::groth16::Proof<E>,
    pub reduction_count: ReductionCount,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Claim<F: LurkField> {
    Evaluation(Evaluation),
    #[serde(bound(
        serialize = "Opening<F>: Serialize",
        deserialize = "Opening<F>: Deserialize<'de>"
    ))]
    Opening(Opening<F>),
}

// This is just a rough idea, mostly here so we can plumb it elsewhere. The idea is that a verifier can sign an
// attestation that a given claim's proof was verified. It motivates the use of an online verifier for demo purposes.
// Although real proofs should be fast to verify, they will still be large relative to a small (auditable) bundle like
// this. Even if not entirely realistic, something with this general *shape* is likely to play a role in a recursive
// system where the ability to aggregate proof verification more soundly is possible.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Cert {
    #[serde(serialize_with = "cid_string", deserialize_with = "string_cid")]
    pub claim_cid: Cid,
    #[serde(serialize_with = "cid_string", deserialize_with = "string_cid")]
    pub proof_cid: Cid,
    pub verified: bool,
    pub verifier_id: String,
    pub signature: String,
}

fn cid_string<S>(c: &Cid, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(&c.to_string())
}

fn string_cid<'de, D>(d: D) -> Result<Cid, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    let string = String::deserialize(d)?;

    Cid::from_str(&string).map_err(|e| D::Error::custom(e.to_string()))
}

#[allow(dead_code)]
impl<F: LurkField> Claim<F> {
    pub fn is_evaluation(&self) -> bool {
        self.evaluation().is_some()
    }
    pub fn is_opening(&self) -> bool {
        self.opening().is_some()
    }
    pub fn evaluation(&self) -> Option<Evaluation> {
        match self {
            Self::Evaluation(e) => Some(e.clone()),
            _ => None,
        }
    }
    pub fn opening(&self) -> Option<Opening<F>> {
        match self {
            Self::Opening(o) => Some(o.clone()),
            _ => None,
        }
    }
}

type E = Error;
impl TryFrom<usize> for ReductionCount {
    type Error = E;

    fn try_from(count: usize) -> Result<Self, <Self as TryFrom<usize>>::Error> {
        match count {
            1 => Ok(ReductionCount::One),
            5 => Ok(ReductionCount::Five),
            10 => Ok(ReductionCount::Ten),
            c => Err(Error::UnsupportedReductionCount(c)),
        }
    }
}
impl ReductionCount {
    fn reduction_frame_count(&self) -> usize {
        match self {
            Self::One => 1,
            Self::Five => 5,
            Self::Ten => 10,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    VerificationError(String),
    UnsupportedReductionCount(usize),
    IOError(io::Error),
    JsonError(serde_json::Error),
    SynthesisError(SynthesisError),
    CommitmentParseError(hex::FromHexError),
    UnknownCommitment,
    OpeningFailure,
    EvaluationFailure,
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::IOError(err)
    }
}
impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Error {
        Error::JsonError(err)
    }
}
impl From<SynthesisError> for Error {
    fn from(err: SynthesisError) -> Error {
        Error::SynthesisError(err)
    }
}

pub trait Id
where
    Self: Sized,
{
    fn id(&self) -> String;
    fn cid(&self) -> Cid;
    fn has_id(&self, id: String) -> bool;
}

pub trait Key<T: ToString>
where
    Self: Sized,
{
    fn key(&self) -> T;
}

impl<T: Serialize> Id for T
where
    for<'de> T: Deserialize<'de>,
{
    fn cid(&self) -> Cid {
        let ipld = to_ipld(self).unwrap();
        let dag_json = DagJsonCodec.encode(&ipld).unwrap();

        let digest = Code::Blake3_256.digest(&dag_json);
        Cid::new_v1(0x55, digest)
    }

    fn id(&self) -> String {
        self.cid().to_string()
    }

    fn has_id(&self, id: String) -> bool {
        self.id() == id
    }
}

pub trait FileStore
where
    Self: Sized,
{
    fn write_to_path<P: AsRef<Path>>(&self, path: P);
    fn read_from_path<P: AsRef<Path>>(path: P) -> Result<Self, Error>;
    fn read_from_stdin() -> Result<Self, Error>;
}

impl<T: Serialize> FileStore for T
where
    for<'de> T: Deserialize<'de>, // + Decode<DagJsonCodec>,
{
    fn write_to_path<P: AsRef<Path>>(&self, path: P) {
        let file = File::create(path).expect("failed to create file");
        let writer = BufWriter::new(&file);

        serde_json::to_writer(writer, &self).expect("failed to write file");
    }

    fn read_from_path<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader).expect("failed to read file"))
    }

    fn read_from_stdin() -> Result<Self, Error> {
        let reader = BufReader::new(io::stdin());
        Ok(serde_json::from_reader(reader).expect("failed to read from stdin"))
    }
}

impl Evaluation {
    fn new<F: LurkField>(
        s: &mut Store<F>,
        input: IO<F>,
        output: IO<F>,
        iterations: Option<usize>, // This might be padded, so is not quite 'iterations' in the sense of number of actual reduction steps required
                                   // to evaluate.
    ) -> Self {
        let status: Status = output.cont.into();
        let terminal = status.is_terminal();

        // For now, conservatively hide all outputs unless output is terminal. TODO: let evaluator configure this in a
        // more fine-grained way, including no hiding.
        // NOTE: If anything is hidden, a proof won't be possible.
        macro_rules! maybe_hide {
            ($x:expr) => {
                if terminal {
                    $x
                } else {
                    "".to_string()
                }
            };
        }

        let expr = input.expr.fmt_to_string(s);
        let env = input.env.fmt_to_string(s);
        let cont = input.cont.fmt_to_string(s);

        let expr_out = maybe_hide!(output.expr.fmt_to_string(s));
        let env_out = maybe_hide!(output.env.fmt_to_string(s));
        let cont_out = maybe_hide!(output.cont.fmt_to_string(s));

        Self {
            expr,
            env,
            cont,
            expr_out,
            env_out,
            cont_out,
            status,
            iterations,
        }
    }

    pub fn eval<F: LurkField + Serialize>(
        store: &mut Store<F>,
        expr: Ptr<F>,
        limit: usize,
    ) -> Self {
        let env = empty_sym_env(store);
        let mut evaluator = Evaluator::new(expr, env, store, limit);

        let input = evaluator.initial();

        let (output, iterations, _) = evaluator.eval();

        Self::new(store, input, output, Some(iterations))
    }
}

impl<F: LurkField + Serialize + DeserializeOwned> Commitment<F> {
    pub fn from_cons(s: &mut Store<F>, ptr: &Ptr<F>) -> Self {
        let digest = *s.hash_expr(ptr).expect("couldn't hash ptr").value();

        assert_eq!(Tag::Cons, ptr.tag());

        Commitment { comm: digest }
    }

    pub fn ptr(&self, s: &mut Store<F>) -> Ptr<F> {
        s.intern_opaque_cons(self.comm)
    }

    pub fn from_ptr_with_hiding(s: &mut Store<F>, function_ptr: &Ptr<F>) -> (Self, F) {
        let secret = F::random(OsRng);

        let commitment = Self::from_ptr_and_secret(s, function_ptr, secret);

        (commitment, secret)
    }

    pub fn from_ptr_and_secret(s: &mut Store<F>, function_ptr: &Ptr<F>, secret: F) -> Self {
        let secret_num = Num::from_scalar(secret);
        let secret_ptr = s.intern_num(secret_num);

        let hidden = s.cons(secret_ptr, *function_ptr);

        Self::from_cons(s, &hidden)
    }

    // Importantly, this ensures the function and secret are in the Store, s.
    fn construct_with_fun_application(
        s: &mut Store<F>,
        function: Function<F>,
        input: Ptr<F>,
        limit: usize,
    ) -> (Self, Ptr<F>) {
        let fun_ptr = function.fun_ptr(s, limit);
        let secret = function.secret.expect("Function secret missing");

        let commitment = Self::from_ptr_and_secret(s, &fun_ptr, secret);

        let cdr = s.sym("cdr");
        let quote = s.sym("quote");

        let secret_ptr = s.num(Num::from_scalar(secret));
        let comm_ptr = s.cons(secret_ptr, fun_ptr);

        let quoted_comm_ptr = s.list(&[quote, comm_ptr]);

        // (cdr <commitment>)
        let fun_expr = s.list(&[cdr, quoted_comm_ptr]);

        // ((cdr <commitment>) input)
        let expression = s.list(&[fun_expr, input]);

        (commitment, expression)
    }

    fn fun_application(&self, s: &mut Store<F>, input: Ptr<F>) -> Ptr<F> {
        let cdr = s.sym("cdr");
        let quote = s.sym("quote");

        let comm_ptr = self.ptr(s);
        let quoted_comm_ptr = s.list(&[quote, comm_ptr]);

        // <commitment> is (secret . fun-expr)

        // (cdr <commitment>)
        let fun_expr = s.list(&[cdr, quoted_comm_ptr]);

        // ((cdr commitment) input)
        s.list(&[fun_expr, input])
    }
}

impl<F: LurkField + Serialize + DeserializeOwned> Function<F> {
    pub fn fun_ptr(&self, s: &mut Store<F>, limit: usize) -> Ptr<F> {
        let source_ptr = self.fun.ptr(s);

        // Evaluate the source to get an actual function.
        let (output, _iterations) = evaluate(s, source_ptr, limit);
        // TODO: Verify that result actually is a function.

        output.expr
    }
}

impl LurkPtr {
    pub fn ptr<F: LurkField + Serialize + DeserializeOwned>(&self, s: &mut Store<F>) -> Ptr<F> {
        match self {
            LurkPtr::Source(source) => s.read(source).expect("could not read source"),
            LurkPtr::ScalarBytes(lurk_scalar_bytes) => {
                let scalar_store: Ipld = DagCborCodec
                    .decode(&lurk_scalar_bytes.scalar_store)
                    .expect("could not read opaque scalar store");
                let scalar_ptr: Ipld = DagCborCodec
                    .decode(&lurk_scalar_bytes.scalar_ptr)
                    .expect("could not read opaque scalar ptr");

                let lurk_ptr = LurkPtr::Ipld(LurkScalarIpld {
                    scalar_store,
                    scalar_ptr,
                });

                lurk_ptr.ptr(s)
            }
            LurkPtr::Ipld(lurk_scalar_ipld) => {
                // FIXME: put the scalar_store in a new field for the store.
                let fun_scalar_store: ScalarStore<F> =
                    from_ipld(lurk_scalar_ipld.scalar_store.clone()).unwrap();
                let fun_scalar_ptr: ScalarPtr<F> =
                    from_ipld(lurk_scalar_ipld.scalar_ptr.clone()).unwrap();
                s.intern_scalar_ptr(fun_scalar_ptr, &fun_scalar_store)
                    .expect("failed to intern scalar_ptr for fun")
            }
        }
    }
}

impl Expression {
    pub fn eval<F: LurkField + Serialize + DeserializeOwned>(
        &self,
        s: &mut Store<F>,
        limit: usize,
    ) -> Result<Ptr<F>, Error> {
        let expr = self.expr.ptr(s);
        let (io, _iterations) = evaluate(s, expr, limit);

        Ok(io.expr)
    }
}

impl Opening<Scalar> {
    pub fn apply_and_prove(
        s: &mut Store<Scalar>,
        input: Ptr<Scalar>,
        function: Function<Scalar>,
        limit: usize,
        chain: bool,
        only_use_cached_proofs: bool,
    ) -> Result<Proof<Bls12>, Error> {
        let claim = Self::apply(s, input, function, limit, chain)?;

        Proof::prove_claim(s, claim, limit, only_use_cached_proofs)
    }

    pub fn open_and_prove(
        s: &mut Store<Scalar>,
        request: OpeningRequest<Scalar>,
        limit: usize,
        only_use_cached_proofs: bool,
    ) -> Result<Proof<Bls12>, Error> {
        let input = request.input.expr.ptr(s);
        let commitment = request.commitment;

        let function_map = committed_function_store();
        let function = function_map
            .get(commitment)
            .ok_or(Error::UnknownCommitment)?;

        Self::apply_and_prove(
            s,
            input,
            function,
            limit,
            request.chain,
            only_use_cached_proofs,
        )
    }

    pub fn open(
        s: &mut Store<Scalar>,
        request: OpeningRequest<Scalar>,
        limit: usize,
        chain: bool,
    ) -> Result<Claim<Scalar>, Error> {
        let input = request.input.expr.ptr(s);
        let commitment = request.commitment;

        let function_map = committed_function_store();
        let function = function_map
            .get(commitment)
            .ok_or(Error::UnknownCommitment)?;

        Self::apply(s, input, function, limit, chain)
    }

    fn _is_chained(&self) -> bool {
        self.new_commitment.is_some()
    }

    fn public_output_expression(&self, s: &mut Store<Scalar>) -> Ptr<Scalar> {
        let result = s.read(&self.output).expect("unreadable result");

        if let Some(commitment) = self.new_commitment {
            let c = commitment.ptr(s);

            s.cons(result, c)
        } else {
            result
        }
    }

    pub fn apply(
        s: &mut Store<Scalar>,
        input: Ptr<Scalar>,
        function: Function<Scalar>,
        limit: usize,
        chain: bool,
    ) -> Result<Claim<Scalar>, Error> {
        let function_map = committed_function_store();

        let (commitment, expression) =
            Commitment::construct_with_fun_application(s, function, input, limit);
        let (public_output, _iterations) = evaluate(s, expression, limit);

        let (new_commitment, output_expr) = if chain {
            // public_output = (result_expr (secret . new_fun))
            let cons = public_output.expr;
            let result_expr = s.car(&cons);
            let new_comm = s.cdr(&cons);
            let new_secret = *s
                .get_expr_hash(&s.car(&new_comm))
                .expect("secret missing")
                .value();

            let new_fun = s.cdr(&new_comm);
            let new_commitment = Commitment::from_cons(s, &new_comm);

            s.hydrate_scalar_cache();
            let (scalar_store, scalar_ptr) = ScalarStore::new_with_expr(s, &new_fun);
            let scalar_ptr = scalar_ptr.unwrap();

            let scalar_store_ipld = to_ipld(scalar_store).unwrap();
            let new_fun_ipld = to_ipld(scalar_ptr).unwrap();

            let scalar_store_bytes = DagCborCodec.encode(&scalar_store_ipld).unwrap();
            let new_fun_bytes = DagCborCodec.encode(&new_fun_ipld).unwrap();

            let again = from_ipld(new_fun_ipld).unwrap();
            assert_eq!(&scalar_ptr, &again);

            // TODO: Can this be made to work?
            // let new_function = Function::<Scalar> {
            //     fun: LurkPtr::Ipld(LurkScalarIpld {
            //         scalar_store: scalar_store_ipld,
            //         scalar_ptr: new_fun_ipld,
            //     }),
            //     secret: Some(new_secret),
            //     commitment: Some(new_commitment),
            // };

            let new_function = Function::<Scalar> {
                fun: LurkPtr::ScalarBytes(LurkScalarBytes {
                    scalar_store: scalar_store_bytes,
                    scalar_ptr: new_fun_bytes,
                }),
                secret: Some(new_secret),
                commitment: Some(new_commitment),
            };

            function_map.set(new_commitment, &new_function)?;

            (Some(new_commitment), result_expr)
        } else {
            (None, public_output.expr)
        };

        let input_string = input.fmt_to_string(s);
        let status = public_output.status();
        let output_string = if status.is_terminal() {
            // Only actual output if result is terminal.
            output_expr.fmt_to_string(s)
        } else {
            // We don't want to leak any internal information in the case of incomplete computations.
            // Provers might want to expose results in the case of explicit errors.
            // For now, don't -- but consider allowing it as an option.
            "".to_string()
        };

        let claim = Claim::Opening(Opening {
            commitment,
            new_commitment,
            input: input_string,
            output: output_string,
            status,
        });

        Ok(claim)
    }
}

impl Proof<Bls12> {
    pub fn eval_and_prove(
        s: &mut Store<Scalar>,
        expr: Ptr<Scalar>,
        limit: usize,
        only_use_cached_proofs: bool,
    ) -> Result<Self, Error> {
        let env = empty_sym_env(s);
        let cont = s.intern_cont_outermost();
        let input = IO { expr, env, cont };

        let (public_output, _iterations) = evaluate(s, expr, limit);
        let evaluation = Evaluation::new(s, input, public_output, None);
        let claim = Claim::Evaluation(evaluation);

        Self::prove_claim(s, claim, limit, only_use_cached_proofs)
    }

    pub fn prove_claim(
        s: &mut Store<Scalar>,
        claim: Claim<Scalar>,
        limit: usize,
        only_use_cached_proofs: bool,
    ) -> Result<Self, Error> {
        let rng = OsRng;
        let proof_map = bls12_proof_cache();
        let function_map = committed_function_store();

        if let Some(proof) = proof_map.get(claim.cid()) {
            return Ok(proof);
        }

        if only_use_cached_proofs {
            // FIXME: Error handling.
            panic!("no cached proof");
        }

        let reduction_count = DEFAULT_REDUCTION_COUNT;

        info!("Getting Parameters");
        let groth_prover = Groth16Prover::new(reduction_count.reduction_frame_count());
        let groth_params = groth_prover.groth_params().unwrap();

        info!("Starting Proving");

        let (expr, env) = match &claim {
            Claim::Evaluation(e) => (
                s.read(&e.expr).expect("bad expression"),
                s.read(&e.env).expect("bad env"),
            ),
            Claim::Opening(o) => {
                let commitment = o.commitment;

                // In order to prove the opening, we need access to the original function.
                let function = function_map
                    .get(commitment)
                    .expect("function for commitment missing");

                let input = s.read(&o.input).expect("bad expression");
                let (c, expression) =
                    Commitment::construct_with_fun_application(s, function, input, limit);

                assert_eq!(commitment, c);

                (expression, empty_sym_env(s))
            }
        };

        let (groth_proof, _public_input, public_output) = groth_prover
            .outer_prove(groth_params, &INNER_PRODUCT_SRS, expr, env, s, limit, rng)
            .expect("Groth proving failed");
        assert!(public_output.is_complete());

        let proof = Proof {
            claim: claim.clone(),
            reduction_count,
            proof: groth_proof,
        };

        match &proof.claim {
            Claim::Opening(o) => {
                if o.status != Status::Terminal {
                    return Err(Error::OpeningFailure);
                };
            }
            Claim::Evaluation(e) => {
                if e.status != Status::Terminal {
                    return Err(Error::EvaluationFailure);
                };
            }
        };

        let verification_result = proof.verify()?;
        assert!(verification_result.verified);

        proof_map.set(claim.cid(), &proof).unwrap();

        Ok(proof)
    }

    pub fn verify(&self) -> Result<VerificationResult, Error> {
        let (public_inputs, public_outputs) = match self.claim {
            Claim::Evaluation(_) => self.verify_evaluation(),
            Claim::Opening(_) => self.verify_opening(),
        }?;
        let mut rng = OsRng;

        info!("Getting Parameters");

        let count = self.proof.proof_count;
        let rc = self.reduction_count;
        let pvk = get_pvk(rc);

        info!("Specializing SRS for {} sub-proofs.", count);
        let srs_vk = INNER_PRODUCT_SRS.specialize_vk(count);
        info!("Starting Verification");

        let verified = Groth16Prover::verify(
            &pvk,
            &srs_vk,
            &public_inputs,
            &public_outputs,
            &self.proof.proof,
            &mut rng,
        )
        .expect("error verifying");

        let result = VerificationResult::new(verified);

        Ok(result)
    }

    pub fn verify_evaluation(&self) -> Result<(Vec<Scalar>, Vec<Scalar>), Error> {
        let mut s = Store::<Scalar>::default();

        let evaluation = &self.claim.evaluation().expect("expected evaluation claim");

        let input_io = {
            let expr = s
                .read(&evaluation.expr)
                .ok_or_else(|| Error::VerificationError("failed to read expr".into()))?;
            let env = s
                .read(&evaluation.env)
                .ok_or_else(|| Error::VerificationError("failed to read env".into()))?;

            // FIXME: We ignore cont and assume Outermost, since we can't read a Cont.
            let cont = s.intern_cont_outermost();

            IO::<Scalar> { expr, env, cont }
        };

        let public_inputs = input_io.to_inputs(&s);

        let output_io = {
            let expr = s
                .read(&evaluation.expr_out)
                .ok_or_else(|| Error::VerificationError("failed to read expr_out".into()))?;
            let env = s.read(&evaluation.env_out).expect("failed to read env_out");
            let cont = evaluation
                .status
                .to_cont(&mut s)
                .ok_or_else(|| Error::VerificationError("continuation cannot be proved".into()))?;

            IO::<Scalar> { expr, env, cont }
        };

        let public_outputs = output_io.to_inputs(&s);

        Ok((public_inputs, public_outputs))
    }

    pub fn verify_opening(&self) -> Result<(Vec<Scalar>, Vec<Scalar>), Error> {
        let mut s = Store::<Scalar>::default();

        assert!(self.claim.is_opening());

        let opening = self.claim.opening().expect("expected opening claim");
        let output = opening.public_output_expression(&mut s);

        let input = s.read(&opening.input).expect("could not read input");

        let expression = opening.commitment.fun_application(&mut s, input);

        let expr = s
            .hash_expr(&expression)
            .expect("failed to hash input expression");

        let empty_env = s.hash_expr(&empty_sym_env(&s)).unwrap();
        let outermost = s.intern_cont_outermost();
        let cont = s.hash_cont(&outermost).unwrap();

        let public_inputs = vec![
            *expr.tag(),
            *expr.value(),
            *empty_env.tag(),
            *empty_env.value(),
            *cont.tag(),
            *cont.value(),
        ];

        let output_io = IO::<Scalar> {
            expr: output,
            env: empty_sym_env(&s),
            cont: s.intern_cont_terminal(),
        };

        let public_outputs = output_io.to_inputs(&s);

        Ok((public_inputs, public_outputs))
    }
}

impl Key<Commitment<Scalar>> for Function<Scalar> {
    fn key(&self) -> Commitment<Scalar> {
        self.commitment.expect("commitment missing")
    }
}

impl Key<Cid> for Proof<Bls12> {
    fn key(&self) -> Cid {
        self.claim.cid()
    }
}

impl VerificationResult {
    fn new(verified: bool) -> Self {
        Self { verified }
    }
}

pub fn evaluate<F: LurkField>(store: &mut Store<F>, expr: Ptr<F>, limit: usize) -> (IO<F>, usize) {
    let env = empty_sym_env(store);
    let mut evaluator = Evaluator::new(expr, env, store, limit);

    let (io, iterations, _) = evaluator.eval();

    assert!(io.is_terminal());
    (io, iterations)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cert_serialization() {
        use serde_json::json;

        let c = Commitment {
            comm: Scalar::from(123),
        };

        let cid = c.cid();
        let cert = Cert {
            claim_cid: cid,
            proof_cid: cid,
            verified: true,
            verifier_id: "asdf".to_string(),
            signature: "fdsa".to_string(),
        };
        let json = json!(cert);

        let string = json.to_string();

        let cert_again: Cert = serde_json::from_str(&string).unwrap();
        assert_eq!(cert, cert_again);
    }
}
