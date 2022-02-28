use bellperson::{
    groth16::{
        self,
        aggregate::{
            aggregate_proofs, aggregate_proofs_and_instances, setup_fake_srs, AggregateProof,
            AggregateProofAndInstance, GenericSRS,
        },
        verify_proof,
    },
    SynthesisError,
};
use blstrs::{Bls12, Scalar};
use ff::PrimeField;
use once_cell::sync::{Lazy, OnceCell};
use pairing_lib::{Engine, MultiMillerLoop};
use rand::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;

use crate::circuit::MultiFrame;
use crate::eval::{Evaluator, Witness, IO};
use crate::proof::{Provable, Prover};
use crate::store::{Ptr, Store};

use std::env;
use std::fs::File;
use std::io;
use std::marker::PhantomData;

const DUMMY_RNG_SEED: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

pub static INNER_PRODUCT_SRS: Lazy<GenericSRS<Bls12>> = Lazy::new(|| load_srs().unwrap());
static FRAME_GROTH_PARAMS: OnceCell<groth16::Parameters<Bls12>> = OnceCell::new();
const MAX_FAKE_SRS_SIZE: usize = 2 << 20;

const TRANSCRIPT_INCLUDE: &[u8] = "LURK-CIRCUIT".as_bytes();

// If you don't have a real SnarkPack SRS symlinked, generate a fake one.
// Don't use this in production!
const FALLBACK_TO_FAKE_SRS: bool = true;

type FrameProofs<'a, E> = (Proof<'a, E>, Vec<Vec<<E as Engine>::Fr>>);

fn load_srs() -> Result<GenericSRS<Bls12>, io::Error> {
    let path = env::current_dir()?.join("params/v28-fil-inner-product-v1.srs");
    let mut rng = rand::thread_rng();
    let f = File::open(path);

    match f {
        Ok(mut f) => GenericSRS::read(&mut f),
        Err(e) => {
            if FALLBACK_TO_FAKE_SRS {
                Ok(setup_fake_srs::<Bls12, _>(&mut rng, MAX_FAKE_SRS_SIZE))
            } else {
                Err(e)
            }
        }
    }
}

#[allow(dead_code)]
pub struct Proof<'a, E: Engine + MultiMillerLoop>
where
    <E as Engine>::Gt: blstrs::Compress,
{
    multiframe_proofs: Vec<(
        MultiFrame<'a, E::Fr, IO<E::Fr>, Witness<E::Fr>>,
        groth16::Proof<E>,
    )>,
    aggregated_proof_and_instance: AggregateProofAndInstance<E>,
    aggregated_proof: AggregateProof<E>,
}

pub trait Groth16<F: PrimeField>: Prover<F>
where
    <Self::E as Engine>::Gt: blstrs::Compress + serde::Serialize,
    <Self::E as Engine>::G1: serde::Serialize,
    <Self::E as Engine>::G1Affine: serde::Serialize,
    <Self::E as Engine>::G2Affine: serde::Serialize,
    <Self::E as Engine>::Fr: serde::Serialize,
{
    type E: Engine + MultiMillerLoop;

    fn groth_params(&self) -> Result<&'static groth16::Parameters<Bls12>, SynthesisError> {
        let store = Store::default();
        // FIXME: Why can't we use this?
        //let multi_frame = self.blank_multi_frame(&store);
        let multi_frame = MultiFrame::blank(&store, self.chunk_frame_count());
        let params = FRAME_GROTH_PARAMS.get_or_try_init::<_, SynthesisError>(|| {
            let rng = &mut XorShiftRng::from_seed(DUMMY_RNG_SEED);
            let params = groth16::generate_random_parameters::<Bls12, _, _>(multi_frame, rng)?;
            Ok(params)
        })?;
        Ok(params)
    }

    fn prove<R: RngCore>(
        &self,
        multi_frame: MultiFrame<
            '_,
            <Self::E as Engine>::Fr,
            IO<<Self::E as Engine>::Fr>,
            Witness<<Self::E as Engine>::Fr>,
        >,
        params: Option<&groth16::Parameters<Self::E>>,
        mut rng: R,
    ) -> Result<groth16::Proof<Self::E>, SynthesisError> {
        self.generate_groth16_proof(multi_frame, params, &mut rng)
    }

    #[allow(clippy::too_many_arguments)]
    fn outer_prove<'a, R: RngCore + Clone>(
        &self,
        params: &groth16::Parameters<Self::E>,
        srs: &GenericSRS<Self::E>,
        expr: Ptr<<Self::E as Engine>::Fr>,
        env: Ptr<<Self::E as Engine>::Fr>,
        store: &'a mut Store<<Self::E as Engine>::Fr>,
        limit: usize,
        mut rng: R,
    ) -> Result<FrameProofs<'a, Self::E>, SynthesisError> {
        let padding_predicate =
            |count, is_terminal: bool| !is_terminal || self.needs_frame_padding(count, is_terminal);
        let frames = Evaluator::generate_frames(expr, env, store, limit, padding_predicate);

        let multiframes = MultiFrame::from_frames(self.chunk_frame_count(), &frames, store);
        let mut proofs = Vec::with_capacity(multiframes.len());
        let mut statements = Vec::with_capacity(multiframes.len());

        // NOTE: frame_proofs are not really needed, but having them helps with
        // testing and building confidence as we work up to fully succinct proofs.
        // Once these are removed a lot of the cloning and awkwardness of assembling
        // results here can be eliminated.
        let multiframes_count = multiframes.len();
        let mut multiframe_proofs = Vec::with_capacity(multiframes_count);

        store.hydrate_scalar_cache();

        let last_multiframe = multiframes.last().unwrap().clone();
        for multiframe in multiframes.into_iter() {
            statements.push(multiframe.public_inputs());
            let proof = self
                .generate_groth16_proof(multiframe.clone(), Some(params), &mut rng)
                .unwrap();

            proofs.push(proof.clone());
            multiframe_proofs.push((multiframe, proof));
        }

        if proofs.len().count_ones() != 1 || proofs.len() < 2 {
            let dummy_multiframe = MultiFrame::make_dummy(
                self.chunk_frame_count(),
                last_multiframe.frames.and_then(|x| x.last().copied()),
                store,
            );

            let dummy_proof = self
                .generate_groth16_proof(dummy_multiframe.clone(), Some(params), &mut rng)
                .unwrap();

            let dummy_statement = dummy_multiframe.public_inputs();
            while proofs.len().count_ones() != 1 || proofs.len() < 2 {
                // Pad proofs and frames to a power of 2.
                proofs.push(dummy_proof.clone());
                statements.push(dummy_statement.clone());
            }
        }
        assert_eq!(1, statements.len().count_ones());
        dbg!(&statements);

        let srs = srs.specialize(proofs.len()).0;

        let aggregated_proof = aggregate_proofs(&srs, TRANSCRIPT_INCLUDE, proofs.as_slice())?;

        let aggregated_proof_and_instance = aggregate_proofs_and_instances(
            &srs,
            TRANSCRIPT_INCLUDE,
            statements.as_slice(),
            proofs.as_slice(),
        )?;

        Ok((
            Proof {
                multiframe_proofs,
                aggregated_proof_and_instance,
                aggregated_proof,
            },
            statements,
        ))
    }

    fn generate_groth16_proof<R: RngCore>(
        &self,
        multi_frame: MultiFrame<
            '_,
            <Self::E as Engine>::Fr,
            IO<<Self::E as Engine>::Fr>,
            Witness<<Self::E as Engine>::Fr>,
        >,
        groth_params: Option<&groth16::Parameters<Self::E>>,
        rng: &mut R,
    ) -> Result<groth16::Proof<Self::E>, SynthesisError>;

    fn verify_groth16_proof(
        // multiframe need not have inner frames populated for verification purposes.
        multiframe: MultiFrame<
            '_,
            <Self::E as Engine>::Fr,
            IO<<Self::E as Engine>::Fr>,
            Witness<<Self::E as Engine>::Fr>,
        >,
        pvk: &groth16::PreparedVerifyingKey<Self::E>,
        proof: groth16::Proof<Self::E>,
    ) -> Result<bool, SynthesisError> {
        let inputs = multiframe.public_inputs();

        verify_proof(pvk, &proof, &inputs)
    }
}

pub struct Groth16Prover<F: PrimeField> {
    chunk_frame_count: usize,
    _p: PhantomData<F>,
}

impl<F: PrimeField> Groth16Prover<F> {
    pub fn new(chunk_frame_count: usize) -> Self {
        Groth16Prover::<F> {
            chunk_frame_count,
            _p: PhantomData::<F>,
        }
    }
}

impl Prover<<Bls12 as Engine>::Fr> for Groth16Prover<<Bls12 as Engine>::Fr> {
    fn chunk_frame_count(&self) -> usize {
        self.chunk_frame_count
    }
}

impl Groth16<<Bls12 as Engine>::Fr> for Groth16Prover<<Bls12 as Engine>::Fr> {
    type E = Bls12;

    fn generate_groth16_proof<R: RngCore>(
        &self,
        multiframe: MultiFrame<
            '_,
            <Self::E as Engine>::Fr,
            IO<<Self::E as Engine>::Fr>,
            Witness<<Self::E as Engine>::Fr>,
        >,
        groth_params: Option<&groth16::Parameters<Self::E>>,
        rng: &mut R,
    ) -> Result<groth16::Proof<Self::E>, SynthesisError> {
        let create_proof = |p| groth16::create_random_proof(multiframe, p, rng);

        if let Some(params) = groth_params {
            let proof = create_proof(params)?;
            Ok(proof)
        } else {
            create_proof(self.groth_params()?)
        }
    }
}

impl
    MultiFrame<'_, <Bls12 as Engine>::Fr, IO<<Bls12 as Engine>::Fr>, Witness<<Bls12 as Engine>::Fr>>
{
    pub fn verify_groth16_proof(
        self,
        pvk: &groth16::PreparedVerifyingKey<Bls12>,
        proof: groth16::Proof<Bls12>,
    ) -> Result<bool, SynthesisError> {
        let inputs: Vec<Scalar> = self.public_inputs();
        verify_proof(pvk, &proof, inputs.as_slice())
    }
}

#[allow(dead_code)]
fn verify_sequential_groth16_proofs(
    multiframe_proofs: Vec<(
        MultiFrame<'_, Scalar, IO<Scalar>, Witness<Scalar>>,
        groth16::Proof<Bls12>,
    )>,
    vk: &groth16::VerifyingKey<Bls12>,
) -> Result<bool, SynthesisError> {
    let pvk = groth16::prepare_verifying_key(vk);

    for (i, (multiframe, proof)) in multiframe_proofs.iter().enumerate() {
        if i > 0 {
            let prev = &multiframe_proofs[i - 1].0;

            if !prev.precedes(multiframe) {
                return Ok(false);
            }
        }

        if !multiframe
            .clone()
            .verify_groth16_proof(&pvk, proof.clone())?
        {
            return Ok(false);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ToInputs;
    use crate::eval::empty_sym_env;
    use crate::proof::{verify_sequential_css, SequentialCS};
    use bellperson::{
        groth16::aggregate::{
            verify_aggregate_proof, verify_aggregate_proof_and_aggregate_instances,
        },
        util_cs::{metric_cs::MetricCS, Comparable, Delta},
        Circuit,
    };
    use blstrs::Scalar as Fr;
    use rand::rngs::OsRng;

    const DEFAULT_CHECK_GROTH16: bool = false;
    const DEFAULT_CHUNK_FRAME_COUNT: usize = 5;

    fn outer_prove_aux<Fo: Fn(&'_ mut Store<Fr>) -> Ptr<Fr>>(
        source: &str,
        expected_result: Fo,
        expected_iterations: usize,
        check_groth16: bool,
        check_constraint_systems: bool,
        limit: usize,
        debug: bool,
    ) {
        let rng = OsRng;

        let mut s = Store::default();
        let expected_result = expected_result(&mut s);

        let expr = s.read(source).unwrap();

        let groth_prover = Groth16Prover::new(DEFAULT_CHUNK_FRAME_COUNT);
        let groth_params = groth_prover.groth_params().unwrap();

        let pvk = groth16::prepare_verifying_key(&groth_params.vk);

        let e = empty_sym_env(&s);

        if check_constraint_systems {
            let padding_predicate = |count, is_terminal: bool| {
                !is_terminal || groth_prover.needs_frame_padding(count, is_terminal)
            };
            let frames = Evaluator::generate_frames(expr, e, &mut s, limit, padding_predicate);
            s.hydrate_scalar_cache();

            let multi_frames = MultiFrame::from_frames(DEFAULT_CHUNK_FRAME_COUNT, &frames, &s);

            let cs = groth_prover.outer_synthesize(&multi_frames).unwrap();

            if !debug {
                //assert_eq!(expected_iterations, cs.len());
                assert_eq!(expected_result, cs[cs.len() - 1].0.output.unwrap().expr);
            }

            let constraint_systems_verified = verify_sequential_css::<Scalar>(&cs).unwrap();
            assert!(constraint_systems_verified);

            check_cs_deltas(&cs, limit);
        }

        let proof_results = if check_groth16 {
            Some(
                groth_prover
                    .outer_prove(
                        &groth_params,
                        &INNER_PRODUCT_SRS,
                        expr.clone(),
                        empty_sym_env(&s),
                        &mut s,
                        limit,
                        rng,
                    )
                    .unwrap(),
            )
        } else {
            None
        };

        if let Some((proof, statements)) = proof_results {
            let frame_proofs = proof.multiframe_proofs.clone();
            if !debug {
                assert_eq!(expected_iterations, frame_proofs.len() - 1);
                assert_eq!(
                    expected_result,
                    proof.multiframe_proofs[frame_proofs.len() - 1]
                        .0
                        .output
                        .as_ref()
                        .unwrap()
                        .expr
                );
            }
            let proofs_verified =
                verify_sequential_groth16_proofs(proof.multiframe_proofs, &groth_params.vk)
                    .unwrap();
            assert!(proofs_verified);

            let mid = IO::<Fr>::input_size();
            let public_inputs = &statements[0][..mid];
            let public_outputs = &statements[statements.len() - 1][mid..];

            let srs_vk = INNER_PRODUCT_SRS.specialize(statements.len()).1;
            let aggregate_proof_and_instances_verified =
                verify_aggregate_proof_and_aggregate_instances(
                    &srs_vk,
                    &pvk,
                    rng.clone(),
                    public_inputs,
                    public_outputs,
                    &proof.aggregated_proof_and_instance,
                    &TRANSCRIPT_INCLUDE,
                )
                .unwrap();

            let aggregate_proof_verified = verify_aggregate_proof(
                &srs_vk,
                &pvk,
                rng.clone(),
                &statements,
                &proof.aggregated_proof,
                TRANSCRIPT_INCLUDE,
            )
            .unwrap();

            assert!(aggregate_proof_verified);
            assert!(aggregate_proof_and_instances_verified);
        };
    }

    pub fn check_cs_deltas(
        constraint_systems: &SequentialCS<Fr, IO<Fr>, Witness<Fr>>,
        limit: usize,
    ) -> () {
        let mut cs_blank = MetricCS::<Fr>::new();
        let store = Store::<Fr>::default();
        let blank_frame = MultiFrame::<Scalar, _, _>::blank(&store, DEFAULT_CHUNK_FRAME_COUNT);
        blank_frame
            .synthesize(&mut cs_blank)
            .expect("failed to synthesize");

        for (_, (_frame, cs)) in constraint_systems.iter().take(limit).enumerate() {
            let delta = cs.delta(&cs_blank, true);
            assert!(delta == Delta::Equal);
        }
    }

    #[test]
    fn outer_prove_arithmetic_let() {
        outer_prove_aux(
            &"(let ((a 5)
                     (b 1)
                     (c 2))
                (/ (+ a b) c))",
            |store| store.num(3),
            18,
            DEFAULT_CHECK_GROTH16,
            true,
            128,
            false,
        );
    }

    #[test]
    fn outer_prove_binop() {
        outer_prove_aux(
            &"(+ 1 2)",
            |store| store.num(3),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            128,
            false,
        );
    }

    #[test]
    fn outer_prove_eq() {
        outer_prove_aux(
            &"(eq 5 5)",
            |store| store.t(),
            3,
            true, // Always check Groth16 in at least one test.
            true,
            128,
            false,
        );
    }

    #[test]
    fn outer_prove_num_equal() {
        outer_prove_aux(
            &"(= 5 5)",
            |store| store.t(),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            128,
            false,
        );
        outer_prove_aux(
            &"(= 5 6)",
            |store| store.nil(),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            128,
            false,
        );
    }

    #[test]
    fn outer_prove_if() {
        outer_prove_aux(
            &"(if t 5 6)",
            |store| store.num(5),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            128,
            false,
        );

        outer_prove_aux(
            &"(if t 5 6)",
            |store| store.num(5),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            128,
            false,
        )
    }
    #[test]
    fn outer_prove_if_fully_evaluates() {
        outer_prove_aux(
            &"(if t (+ 5 5) 6)",
            |store| store.num(10),
            5,
            DEFAULT_CHECK_GROTH16,
            true,
            128,
            false,
        );
    }

    #[test]
    #[ignore] // Skip expensive tests in CI for now. Do run these locally, please.
    fn outer_prove_recursion1() {
        outer_prove_aux(
            &"(letrec ((exp (lambda (base)
                               (lambda (exponent)
                                 (if (= 0 exponent)
                                     1
                                     (* base ((exp base) (- exponent 1))))))))
                ((exp 5) 3))",
            |store| store.num(125),
            // 117, // FIXME: is this change correct?
            91,
            DEFAULT_CHECK_GROTH16,
            true,
            256,
            false,
        );
    }

    #[test]
    #[ignore] // Skip expensive tests in CI for now. Do run these locally, please.
    fn outer_prove_recursion2() {
        outer_prove_aux(
            &"(letrec ((exp (lambda (base)
                                  (lambda (exponent)
                                     (lambda (acc)
                                       (if (= 0 exponent)
                                          acc
                                          (((exp base) (- exponent 1)) (* acc base))))))))
                (((exp 5) 5) 1))",
            |store| store.num(3125),
            // 248, // FIXME: is this change correct?
            201,
            DEFAULT_CHECK_GROTH16,
            true,
            256,
            false,
        );
    }
}
