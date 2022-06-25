#![allow(non_snake_case)]
use log::{info, warn};

use std::marker::PhantomData;

use bellperson::{
    util_cs::test_cs::TestConstraintSystem, Circuit, ConstraintSystem, SynthesisError,
};
use merlin::Transcript;
use nova::{
    bellperson::{
        r1cs::{NovaShape, NovaWitness},
        shape_cs::ShapeCS,
        solver::SatisfyingAssignment,
    },
    errors::NovaError,
    r1cs::{
        R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
    },
    traits::Group,
    FinalSNARK, StepSNARK,
};
use pasta_curves::pallas;

use crate::circuit::MultiFrame;
use crate::eval::{Evaluator, Frame, Witness, IO};

use crate::field::LurkField;
use crate::proof::Prover;
use crate::store::{Ptr, Store};

use super::MultiFrameSynthesizer;

type PallasPoint = pallas::Point;

pub struct Proof<G: Group> {
    pub step_proofs: Vec<StepSNARK<G>>,
    pub final_proof: FinalSNARK<G>,
    pub final_instance: RelaxedR1CSInstance<G>,
}

impl<G: Group> Proof<G> {
    pub fn verify(
        &self,
        shape_and_gens: &(R1CSShape<G>, R1CSGens<G>),
        instance: &RelaxedR1CSInstance<G>,
    ) -> bool {
        self.final_proof
            .verify(&shape_and_gens.1, &shape_and_gens.0, instance)
            .is_ok()
    }
}

pub trait Nova<F: LurkField>: Prover<F>
where
    <Self::Grp as Group>::Scalar: ff::PrimeField,
{
    type Grp: Group;

    fn make_r1cs(
        multi_frame: MultiFrame<
            '_,
            <Self::Grp as Group>::Scalar,
            IO<<Self::Grp as Group>::Scalar>,
            Witness<<Self::Grp as Group>::Scalar>,
        >,
        shape: &R1CSShape<Self::Grp>,
        gens: &R1CSGens<Self::Grp>,
    ) -> Result<(R1CSInstance<Self::Grp>, R1CSWitness<Self::Grp>), NovaError>
    where
        <<Self as Nova<F>>::Grp as Group>::Scalar: LurkField,
    {
        let mut cs = SatisfyingAssignment::<Self::Grp>::new();

        multi_frame.synthesize(&mut cs).unwrap();

        let (instance, witness) = cs.r1cs_instance_and_witness(shape, gens)?;

        Ok((instance, witness))
    }
    fn get_evaluation_frames(
        &self,
        expr: Ptr<<Self::Grp as Group>::Scalar>,
        env: Ptr<<Self::Grp as Group>::Scalar>,
        store: &mut Store<<Self::Grp as Group>::Scalar>,
        limit: usize,
    ) -> Vec<Frame<IO<<Self::Grp as Group>::Scalar>, Witness<<Self::Grp as Group>::Scalar>>>
    where
        <<Self as Nova<F>>::Grp as Group>::Scalar: LurkField,
    {
        let padding_predicate = |count| self.needs_frame_padding(count);

        let frames = Evaluator::generate_frames(expr, env, store, limit, padding_predicate);
        store.hydrate_scalar_cache();

        frames
    }
    fn evaluate_and_prove(
        &self,
        expr: Ptr<<Self::Grp as Group>::Scalar>,
        env: Ptr<<Self::Grp as Group>::Scalar>,
        store: &mut Store<<Self::Grp as Group>::Scalar>,
        limit: usize,
    ) -> Result<(Proof<Self::Grp>, RelaxedR1CSInstance<Self::Grp>), SynthesisError>
    where
        <<Self as Nova<F>>::Grp as Group>::Scalar: LurkField,
    {
        let frames = self.get_evaluation_frames(expr, env, store, limit);

        let (shape, gens) = self.make_shape_and_gens();

        self.make_proof(frames.as_slice(), &shape, &gens, store, true)
    }

    fn make_shape_and_gens(&self) -> (R1CSShape<Self::Grp>, R1CSGens<Self::Grp>);

    fn make_proof(
        &self,
        frames: &[Frame<IO<<Self::Grp as Group>::Scalar>, Witness<<Self::Grp as Group>::Scalar>>],
        shape: &R1CSShape<Self::Grp>,
        gens: &R1CSGens<Self::Grp>,
        store: &mut Store<<Self::Grp as Group>::Scalar>,
        verify_steps: bool, // Sanity check for development, until we have recursion.
    ) -> Result<(Proof<Self::Grp>, RelaxedR1CSInstance<Self::Grp>), SynthesisError>
    where
        <<Self as Nova<F>>::Grp as Group>::Scalar: LurkField,
    {
        let multiframes = MultiFrame::from_frames(self.chunk_frame_count(), frames, store);
        for mf in &multiframes {
            assert_eq!(
                Some(self.chunk_frame_count()),
                mf.frames.clone().map(|x| x.len())
            );
        }
        let r1cs_instances = multiframes
            .iter()
            .map(|f| Self::make_r1cs(f.clone(), shape, gens).unwrap())
            .collect::<Vec<_>>();

        let mut step_proofs = Vec::new();
        let mut prover_transcript = Transcript::new(b"LurkProver");
        let mut verifier_transcript = Transcript::new(b"LurkVerifier");

        let initial_acc = (
            RelaxedR1CSInstance::default(gens, shape),
            RelaxedR1CSWitness::default(shape),
        );

        let (acc_U, acc_W) =
            r1cs_instances
                .iter()
                .fold(initial_acc, |(acc_U, acc_W), (next_U, next_W)| {
                    let (step_proof, (step_U, step_W)) = Self::make_step_snark(
                        gens,
                        shape,
                        &acc_U,
                        &acc_W,
                        next_U,
                        next_W,
                        &mut prover_transcript,
                    );
                    if verify_steps {
                        step_proof
                            .verify(&acc_U, next_U, &mut verifier_transcript)
                            .unwrap();
                        step_proofs.push(step_proof);
                    };
                    (step_U, step_W)
                });

        let final_proof = Self::make_final_snark(&acc_W);

        let proof = Proof {
            step_proofs,
            final_proof,
            final_instance: acc_U.clone(),
        };

        Ok((proof, acc_U))
    }

    fn make_step_snark(
        gens: &R1CSGens<Self::Grp>,
        S: &R1CSShape<Self::Grp>,
        r_U: &RelaxedR1CSInstance<Self::Grp>,
        r_W: &RelaxedR1CSWitness<Self::Grp>,
        U2: &R1CSInstance<Self::Grp>,
        W2: &R1CSWitness<Self::Grp>,
        prover_transcript: &mut merlin::Transcript,
    ) -> (
        StepSNARK<Self::Grp>,
        (
            RelaxedR1CSInstance<Self::Grp>,
            RelaxedR1CSWitness<Self::Grp>,
        ),
    ) {
        let res = StepSNARK::prove(gens, S, r_U, r_W, U2, W2, prover_transcript);
        res.expect("make_step_snark failed")
    }

    fn make_final_snark(W: &RelaxedR1CSWitness<Self::Grp>) -> FinalSNARK<Self::Grp> {
        // produce a final SNARK
        let res = FinalSNARK::prove(W);
        res.expect("make_final_snark failed")
    }
}

pub struct NovaProver<F: LurkField> {
    chunk_frame_count: usize,
    _p: PhantomData<F>,
}

impl<F: LurkField> NovaProver<F> {
    pub fn new(chunk_frame_count: usize) -> Self {
        NovaProver::<F> {
            chunk_frame_count,
            _p: PhantomData::<F>,
        }
    }
}

impl<F: LurkField> Prover<F> for NovaProver<F> {
    fn chunk_frame_count(&self) -> usize {
        self.chunk_frame_count
    }
}

impl<F: LurkField> Nova<F> for NovaProver<F> {
    type Grp = PallasPoint;

    fn make_shape_and_gens(&self) -> (R1CSShape<Self::Grp>, R1CSGens<Self::Grp>) {
        let mut cs = ShapeCS::<Self::Grp>::new();
        let store = Store::<<Self::Grp as Group>::Scalar>::default();

        MultiFrame::blank(&store, self.chunk_frame_count)
            .synthesize(&mut cs)
            .unwrap();

        let shape = cs.r1cs_shape();
        let gens = cs.r1cs_gens();

        (shape, gens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::empty_sym_env;
    use crate::proof::{verify_sequential_css, SequentialCS};
    use crate::store::ContPtr;
    use crate::writer::Write;

    use bellperson::util_cs::{metric_cs::MetricCS, Comparable, Delta};
    use pallas::Scalar as Fr;

    const DEFAULT_CHUNK_FRAME_COUNT: usize = 5;

    fn nova_test_aux(
        s: &mut Store<Fr>,
        expr: &str,
        expected_result: Option<Ptr<Fr>>,
        expected_env: Option<Ptr<Fr>>,
        expected_cont: Option<ContPtr<Fr>>,
        expected_emitted: Option<Vec<Ptr<Fr>>>,
        expected_iterations: usize,
    ) {
        nova_test_full_aux(
            s,
            expr,
            expected_result,
            expected_env,
            expected_cont,
            expected_emitted,
            expected_iterations,
            DEFAULT_CHUNK_FRAME_COUNT,
            false,
        )
    }

    fn nova_test_full_aux(
        s: &mut Store<Fr>,
        expr: &str,
        expected_result: Option<Ptr<Fr>>,
        expected_env: Option<Ptr<Fr>>,
        expected_cont: Option<ContPtr<Fr>>,
        expected_emitted: Option<Vec<Ptr<Fr>>>,
        expected_iterations: usize,
        chunk_frame_count: usize,
        check_nova: bool,
    ) {
        let limit = 100000;
        let expr = s.read(expr).unwrap();

        let e = empty_sym_env(&s);

        let nova_prover = NovaProver::<Fr>::new(chunk_frame_count);
        let proof_results = if check_nova {
            Some(
                nova_prover
                    .evaluate_and_prove(expr, empty_sym_env(&s), s, limit)
                    .unwrap(),
            )
        } else {
            None
        };

        if check_nova {
            let shape_and_gens = nova_prover.make_shape_and_gens();
            if let Some((proof, instance)) = proof_results {
                proof.verify(&shape_and_gens, &instance);
            }
        }

        let frames = nova_prover.get_evaluation_frames(expr, e, s, limit);

        let multiframes = MultiFrame::from_frames(nova_prover.chunk_frame_count(), &frames, &s);

        let synthesizer =
            MultiFrameSynthesizer::<'_, _, TestConstraintSystem<Fr>>::from_multiframes(
                &multiframes,
            );
        let len = synthesizer.len();

        let adjusted_iterations = nova_prover.expected_total_iterations(expected_iterations);
        let mut previous_frame: Option<MultiFrame<Fr, IO<Fr>, Witness<Fr>>> = None;

        let mut cs_blank = MetricCS::<Fr>::new();
        let store = Store::<Fr>::default();

        let blank = MultiFrame::<Fr, IO<Fr>, Witness<Fr>>::blank(&store, chunk_frame_count);
        blank
            .synthesize(&mut cs_blank)
            .expect("failed to synthesize blank");

        use crate::proof::Provable;
        for (i, (multiframe, cs)) in synthesizer.enumerate() {
            if let Some(prev) = previous_frame {
                assert!(prev.precedes(&multiframe));
            }
            assert!(cs.is_satisfied());
            assert!(cs.verify(&multiframe.public_inputs()));

            previous_frame = Some(multiframe);

            let delta = cs.delta(&cs_blank, true);
            dbg!(&i, &delta);
            assert!(delta == Delta::Equal);
        }
        let output = previous_frame.unwrap().output.unwrap();

        if let Some(expected_emitted) = expected_emitted {
            let emitted_vec: Vec<_> = frames
                .iter()
                .flat_map(|frame| frame.output.maybe_emitted_expression(&s))
                .collect();
            assert_eq!(expected_emitted, emitted_vec);
        }

        if let Some(expected_result) = expected_result {
            assert_eq!(expected_result, output.expr);
        }
        if let Some(expected_env) = expected_env {
            assert_eq!(expected_env, output.env);
        }
        if let Some(expected_cont) = expected_cont {
            assert_eq!(expected_cont, output.cont);
        } else {
            assert_eq!(s.get_cont_terminal(), output.cont);
        }

        assert_eq!(expected_iterations, Frame::significant_frame_count(&frames));
        assert_eq!(adjusted_iterations, len);
    }

    pub fn check_cs_deltas(
        constraint_systems: &SequentialCS<Fr, IO<Fr>, Witness<Fr>>,
        chunk_frame_count: usize,
    ) {
        let mut cs_blank = MetricCS::<Fr>::new();
        let store = Store::<Fr>::default();

        let blank = MultiFrame::<Fr, IO<Fr>, Witness<Fr>>::blank(&store, chunk_frame_count);
        blank
            .synthesize(&mut cs_blank)
            .expect("failed to synthesize");

        for (i, (_frame, cs)) in constraint_systems.iter().enumerate() {
            let delta = cs.delta(&cs_blank, true);
            dbg!(i, &delta);
            assert!(delta == Delta::Equal);
        }
    }

    pub fn check_cs_delta(cs_blank: MetricCS<Fr>, cs: &TestConstraintSystem<Fr>) {
        let delta = cs.delta(&cs_blank, true);
        dbg!(&delta);
        assert!(delta == Delta::Equal);
    }

    // IMPORTANT: Run next tests at least once. Some are ignored because they
    // are expensive. The criteria is that if the number of iteractions is
    // more than 30 we ignore it.
    ////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_outer_prove_binop() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(3);
        let terminal = s.get_cont_terminal();
        nova_test_aux(s, "(+ 1 2)", Some(expected), None, Some(terminal), None, 3);
    }

    #[test]
    #[ignore]
    fn outer_prove_arithmetic_let() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(3);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((a 5)
                      (b 1)
                      (c 2))
                 (/ (+ a b) c))",
            Some(expected),
            None,
            Some(terminal),
            None,
            18,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_eq() {
        let s = &mut Store::<Fr>::default();
        let expected = s.t();
        let terminal = s.get_cont_terminal();
        nova_test_full_aux(
            s,
            "(eq 5 5)",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
            DEFAULT_CHUNK_FRAME_COUNT,
            true,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_num_equal() {
        let s = &mut Store::<Fr>::default();
        let expected = s.t();
        let terminal = s.get_cont_terminal();
        nova_test_aux(s, "(= 5 5)", Some(expected), None, Some(terminal), None, 3);

        let expected = s.nil();
        let terminal = s.get_cont_terminal();
        nova_test_aux(s, "(= 5 6)", Some(expected), None, Some(terminal), None, 3);
    }

    #[test]
    fn outer_prove_invalid_num_equal() {
        let s = &mut Store::<Fr>::default();
        let expected = s.nil();
        let error = s.get_cont_error();
        nova_test_aux(s, "(= 5 nil)", Some(expected), None, Some(error), None, 3);

        let expected = s.num(5);
        nova_test_aux(s, "(= nil 5)", Some(expected), None, Some(error), None, 3);
    }

    #[test]
    fn outer_prove_quote_end_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(quote (1) (2))", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_if() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(5);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(if t 5 6)",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
        );

        let expected = s.num(6);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(if nil 5 6)",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
        )
    }

    #[test]
    fn outer_prove_if_end_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(5);
        let error = s.get_cont_error();
        nova_test_aux(
            s,
            "(if nil 5 6 7)",
            Some(expected),
            None,
            Some(error),
            None,
            2,
        )
    }

    #[test]
    #[ignore]
    fn outer_prove_if_fully_evaluates() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(10);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(if t (+ 5 5) 6)",
            Some(expected),
            None,
            Some(terminal),
            None,
            5,
        );
    }

    #[test]
    #[ignore] // Skip expensive tests in CI for now. Do run these locally, please.
    fn outer_prove_recursion1() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(25);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((exp (lambda (base)
                               (lambda (exponent)
                                 (if (= 0 exponent)
                                     1
                                     (* base ((exp base) (- exponent 1))))))))
                 ((exp 5) 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            66,
        );
    }

    #[test]
    #[ignore] // Skip expensive tests in CI for now. Do run these locally, please.
    fn outer_prove_recursion2() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(25);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((exp (lambda (base)
                                  (lambda (exponent)
                                     (lambda (acc)
                                       (if (= 0 exponent)
                                          acc
                                          (((exp base) (- exponent 1)) (* acc base))))))))
                (((exp 5) 2) 1))",
            Some(expected),
            None,
            Some(terminal),
            None,
            93,
        );
    }

    fn outer_prove_unop_regression_aux(chunk_count: usize) {
        let s = &mut Store::<Fr>::default();
        let expected = s.sym("t");
        let terminal = s.get_cont_terminal();
        nova_test_full_aux(
            s,
            "(atom 123)",
            Some(expected),
            None,
            Some(terminal),
            None,
            2,
            chunk_count, // This needs to be 1 to exercise the bug.
            false,
        );

        let expected = s.num(1);
        nova_test_full_aux(
            s,
            "(car '(1 . 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            2,
            chunk_count, // This needs to be 1 to exercise the bug.
            false,
        );

        let expected = s.num(2);
        nova_test_full_aux(
            s,
            "(cdr '(1 . 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            2,
            chunk_count, // This needs to be 1 to exercise the bug.
            false,
        );

        let expected = s.num(123);
        nova_test_full_aux(
            s,
            "(emit 123)",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
            chunk_count,
            false,
        )
    }

    #[test]
    #[ignore]
    fn outer_prove_unop_regression() {
        // We need to at least use chunk size 1 to exercise the regression.
        // Also use a non-1 value to check the MultiFrame case.
        for i in 1..2 {
            outer_prove_unop_regression_aux(i);
        }
    }

    #[test]
    #[ignore]
    fn outer_prove_emit_output() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(123);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(emit 123)",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(99);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "((lambda (x) x) 99)",
            Some(expected),
            None,
            Some(terminal),
            None,
            4,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate2() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(99);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "((lambda (y)
                    ((lambda (x) y) 888))
                  99)",
            Some(expected),
            None,
            Some(terminal),
            None,
            9,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate3() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(999);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "((lambda (y)
                     ((lambda (x)
                        ((lambda (z) z)
                         x))
                      y))
                   999)",
            Some(expected),
            None,
            Some(terminal),
            None,
            10,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate4() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(888);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "((lambda (y)
                     ((lambda (x)
                        ((lambda (z) z)
                         x))
                      ;; NOTE: We pass a different value here.
                      888))
                  999)",
            Some(expected),
            None,
            Some(terminal),
            None,
            10,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate5() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(999);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(((lambda (fn)
                      (lambda (x) (fn x)))
                    (lambda (y) y))
                   999)",
            Some(expected),
            None,
            Some(terminal),
            None,
            13,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_sum() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(9);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(+ 2 (+ 3 4))",
            Some(expected),
            None,
            Some(terminal),
            None,
            6,
        );
    }

    #[test]
    fn outer_prove_evaluate_binop_rest_is_nil() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(9);
        let error = s.get_cont_error();
        nova_test_aux(s, "(- 9 8 7)", Some(expected), None, Some(error), None, 2);
    }

    #[test]
    fn outer_prove_evaluate_relop_rest_is_nil() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(9);
        let error = s.get_cont_error();
        nova_test_aux(s, "(= 9 8 7)", Some(expected), None, Some(error), None, 2);
    }

    #[test]
    fn outer_prove_evaluate_diff() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(4);
        let terminal = s.get_cont_terminal();
        nova_test_aux(s, "(- 9 5)", Some(expected), None, Some(terminal), None, 3);
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_product() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(45);
        let terminal = s.get_cont_terminal();
        nova_test_aux(s, "(* 9 5)", Some(expected), None, Some(terminal), None, 3);
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_quotient() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(7);
        let terminal = s.get_cont_terminal();
        nova_test_aux(s, "(/ 21 3)", Some(expected), None, Some(terminal), None, 3);
    }

    #[test]
    fn outer_prove_error_div_by_zero() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(0);
        let error = s.get_cont_error();
        nova_test_aux(s, "(/ 21 0)", Some(expected), None, Some(error), None, 3);
    }

    #[test]
    fn outer_prove_error_invalid_type_and_not_cons() {
        let s = &mut Store::<Fr>::default();
        let expected = s.nil();
        let error = s.get_cont_error();
        nova_test_aux(s, "(/ 21 nil)", Some(expected), None, Some(error), None, 3);
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_adder() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(5);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(((lambda (x)
                    (lambda (y)
                      (+ x y)))
                  2)
                 3)",
            Some(expected),
            None,
            Some(terminal),
            None,
            13,
        );
    }

    #[test]
    fn outer_prove_evaluate_current_env_simple() {
        let s = &mut Store::<Fr>::default();
        let expected = s.nil();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(current-env)",
            Some(expected),
            None,
            Some(terminal),
            None,
            1,
        );
    }

    #[test]
    fn outer_prove_evaluate_current_env_rest_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let expected = s.nil();
        let error = s.get_cont_error();
        nova_test_aux(
            s,
            "(current-env a)",
            Some(expected),
            None,
            Some(error),
            None,
            1,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_let_simple() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(1);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((a 1))
                  a)",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
        );
    }

    #[test]
    fn outer_prove_evaluate_let_end_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(let ((a 1 2)) a)", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_letrec_end_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(letrec ((a 1 2)) a)", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_let_empty_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(let)", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_let_empty_body_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(let ((a 1)))", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_letrec_empty_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(letrec)", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_letrec_empty_body_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(letrec ((a 1)))", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_let_body_nil() {
        let s = &mut Store::<Fr>::default();
        let expected = s.t();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(eq nil (let () nil))",
            Some(expected),
            None,
            Some(terminal),
            None,
            4,
        );
    }

    #[test]
    fn outer_prove_evaluate_let_rest_body_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(let ((a 1)) a 1)", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_letrec_rest_body_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(letrec ((a 1)) a 1)", None, None, Some(error), None, 1);
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_let_null_bindings() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(3);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let () (+ 1 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            4,
        );
    }
    #[test]
    #[ignore]
    fn outer_prove_evaluate_letrec_null_bindings() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(3);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec () (+ 1 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            4,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_let() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(6);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((a 1)
                       (b 2)
                       (c 3))
                  (+ a (+ b c)))",
            Some(expected),
            None,
            Some(terminal),
            None,
            18,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_arithmetic() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(20);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "((((lambda (x)
                      (lambda (y)
                        (lambda (z)
                          (* z
                             (+ x y)))))
                    2)
                  3)
                 4)",
            Some(expected),
            None,
            Some(terminal),
            None,
            23,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_arithmetic_let() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(20);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((x 2)
                        (y 3)
                        (z 4))
                   (* z (+ x y)))",
            Some(expected),
            None,
            Some(terminal),
            None,
            18,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_comparison() {
        let s = &mut Store::<Fr>::default();
        let expected = s.t();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((x 2)
                       (y 3)
                       (z 4))
                  (= 20 (* z
                           (+ x y))))",
            Some(expected),
            None,
            Some(terminal),
            None,
            21,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_conditional() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(5);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((true (lambda (a)
                               (lambda (b)
                                 a)))
                       (false (lambda (a)
                                (lambda (b)
                                  b)))
                      ;; NOTE: We cannot shadow IF because it is built-in.
                      (if- (lambda (a)
                             (lambda (c)
                               (lambda (cond)
                                 ((cond a) c))))))
                 (((if- 5) 6) true))",
            Some(expected),
            None,
            Some(terminal),
            None,
            35,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_conditional2() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(6);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((true (lambda (a)
                               (lambda (b)
                                 a)))
                       (false (lambda (a)
                                (lambda (b)
                                  b)))
                      ;; NOTE: We cannot shadow IF because it is built-in.
                      (if- (lambda (a)
                             (lambda (c)
                               (lambda (cond)
                                 ((cond a) c))))))
                 (((if- 5) 6) false))",
            Some(expected),
            None,
            Some(terminal),
            None,
            32,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_fundamental_conditional_bug() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(5);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((true (lambda (a)
                               (lambda (b)
                                 a)))
                       ;; NOTE: We cannot shadow IF because it is built-in.
                       (if- (lambda (a)
                              (lambda (c)
                               (lambda (cond)
                                 ((cond a) c))))))
                 (((if- 5) 6) true))",
            Some(expected),
            None,
            Some(terminal),
            None,
            32,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_if() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(6);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(if nil 5 6)",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_fully_evaluates() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(10);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(if t (+ 5 5) 6)",
            Some(expected),
            None,
            Some(terminal),
            None,
            5,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_recursion() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(25);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((exp (lambda (base)
                                   (lambda (exponent)
                                     (if (= 0 exponent)
                                         1
                                         (* base ((exp base) (- exponent 1))))))))
                           ((exp 5) 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            66,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_recursion_multiarg() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(25);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((exp (lambda (base exponent)
                                   (if (= 0 exponent)
                                       1
                                       (* base (exp base (- exponent 1)))))))
                           (exp 5 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            69,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_recursion_optimized() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(25);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((exp (lambda (base)
                                (letrec ((base-inner
                                           (lambda (exponent)
                                             (if (= 0 exponent)
                                                 1
                                                 (* base (base-inner (- exponent 1)))))))
                                        base-inner))))
                   ((exp 5) 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            56,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_tail_recursion() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(25);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((exp (lambda (base)
                                   (lambda (exponent-remaining)
                                     (lambda (acc)
                                       (if (= 0 exponent-remaining)
                                           acc
                                           (((exp base) (- exponent-remaining 1)) (* acc base))))))))
                          (((exp 5) 2) 1))",
            Some(expected),
            None,
            Some(terminal),
            None,
            93,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_tail_recursion_somewhat_optimized() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(25);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((exp (lambda (base)
                                   (letrec ((base-inner
                                              (lambda (exponent-remaining)
                                                (lambda (acc)
                                                  (if (= 0 exponent-remaining)
                                                      acc
                                                     ((base-inner (- exponent-remaining 1)) (* acc base)))))))
                                           base-inner))))
                          (((exp 5) 2) 1))",
            Some(expected),
            None,
            Some(terminal),
            None,
            81,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_no_mutual_recursion() {
        let s = &mut Store::<Fr>::default();
        let expected = s.t();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((even (lambda (n)
                                  (if (= 0 n)
                                      t
                                      (odd (- n 1)))))
                          (odd (lambda (n)
                                 (even (- n 1)))))
                        ;; NOTE: This is not true mutual-recursion.
                        ;; However, it exercises the behavior of LETREC.
                        (odd 1))",
            Some(expected),
            None,
            Some(terminal),
            None,
            22,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_no_mutual_recursion_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(
            s,
            "(letrec ((even (lambda (n)
                                  (if (= 0 n)
                                      t
                                      (odd (- n 1)))))
                          (odd (lambda (n)
                                 (even (- n 1)))))
                        ;; NOTE: This is not true mutual-recursion.
                        ;; However, it exercises the behavior of LETREC.
                        (odd 2))",
            None,
            None,
            Some(error),
            None,
            25,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_cons1() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(1);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(car (cons 1 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            5,
        );
    }

    #[test]
    fn outer_prove_evaluate_car_end_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(car (1 2) 3)", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_cdr_end_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(cdr (1 2) 3)", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_atom_end_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(atom 123 4)", None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_evaluate_emit_end_is_nil_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "(emit 123 4)", None, None, Some(error), None, 1);
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_cons2() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(2);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(cdr (cons 1 2))",
            Some(expected),
            None,
            Some(terminal),
            None,
            5,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_zero_arg_lambda1() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(123);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "((lambda () 123))",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_zero_arg_lambda2() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(10);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((x 9) (f (lambda () (+ x 1)))) (f))",
            Some(expected),
            None,
            Some(terminal),
            None,
            10,
        );
    }
    #[test]
    fn outer_prove_evaluate_zero_arg_lambda3() {
        let s = &mut Store::<Fr>::default();
        let expected = {
            let arg = s.sym("x");
            let num = s.num(123);
            let body = s.list(&[num]);
            let env = s.nil();
            s.intern_fun(arg, body, env)
        };
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "((lambda (x) 123))",
            Some(expected),
            None,
            Some(terminal),
            None,
            3,
        );
    }

    #[test]
    fn outer_prove_evaluate_zero_arg_lambda4() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, "((lambda () 123) 1)", None, None, Some(error), None, 3);
    }

    #[test]
    fn outer_prove_evaluate_zero_arg_lambda5() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(123);
        let error = s.get_cont_error();
        nova_test_aux(s, "(123)", Some(expected), None, Some(error), None, 2);
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_minimal_tail_call() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(123);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec
                   ((f (lambda (x)
                         (if (= x 3)
                             123
                             (f (+ x 1))))))
                   (f 0))",
            Some(expected),
            None,
            Some(terminal),
            None,
            50,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_cons_in_function1() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(2);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(((lambda (a)
                    (lambda (b)
                      (car (cons a b))))
                  2)
                 3)",
            Some(expected),
            None,
            Some(terminal),
            None,
            15,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_cons_in_function2() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(3);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(((lambda (a)
                    (lambda (b)
                      (cdr (cons a b))))
                  2)
                 3)",
            Some(expected),
            None,
            Some(terminal),
            None,
            15,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_multiarg_eval_bug() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(2);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(car (cdr '(1 2 3 4)))",
            Some(expected),
            None,
            Some(terminal),
            None,
            4,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_multiple_letrec_bindings() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(123);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec
                   ((x 888)
                    (f (lambda (x)
                         (if (= x 5)
                             123
                             (f (+ x 1))))))
                  (f 0))",
            Some(expected),
            None,
            Some(terminal),
            None,
            78,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_tail_call2() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(123);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec
                   ((f (lambda (x)
                         (if (= x 5)
                             123
                             (f (+ x 1)))))
                    (g (lambda (x) (f x))))
                  (g 0))",
            Some(expected),
            None,
            Some(terminal),
            None,
            84,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_multiple_letrecstar_bindings() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(13);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((double (lambda (x) (* 2 x)))
                           (square (lambda (x) (* x x))))
                          (+ (square 3) (double 2)))",
            Some(expected),
            None,
            Some(terminal),
            None,
            22,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_multiple_letrecstar_bindings_referencing() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(11);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((double (lambda (x) (* 2 x)))
                           (double-inc (lambda (x) (+ 1 (double x)))))
                          (+ (double 3) (double-inc 2)))",
            Some(expected),
            None,
            Some(terminal),
            None,
            31,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_multiple_letrecstar_bindings_recursive() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(33);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((exp (lambda (base exponent)
                                  (if (= 0 exponent)
                                      1
                                      (* base (exp base (- exponent 1))))))
                           (exp2 (lambda (base exponent)
                                   (if (= 0 exponent)
                                      1
                                      (* base (exp2 base (- exponent 1))))))
                          (exp3 (lambda (base exponent)
                                  (if (= 0 exponent)
                                      1
                                      (* base (exp3 base (- exponent 1)))))))
                         (+ (+ (exp 3 2) (exp2 2 3))
                            (exp3 4 2)))",
            Some(expected),
            None,
            Some(terminal),
            None,
            242,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_dont_discard_rest_env() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(18);
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(let ((z 9))
                   (letrec ((a 1)
                             (b 2)
                             (l (lambda (x) (+ z x))))
                            (l 9)))",
            Some(expected),
            None,
            Some(terminal),
            None,
            22,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_fibonacci() {
        let s = &mut Store::<Fr>::default();
        let expected = s.num(1);
        let terminal = s.get_cont_terminal();
        nova_test_full_aux(
            s,
            "(letrec ((next (lambda (a b n target)
                     (if (eq n target)
                         a
                         (next b
                             (+ a b)
                             (+ 1 n)
                            target))))
                    (fib (next 0 1 0)))
                (fib 1))",
            Some(expected),
            None,
            Some(terminal),
            None,
            89,
            5,
            false,
        );
    }

    // #[test]
    // #[ignore]
    // fn outer_prove_evaluate_fibonacci_100() {
    //     let s = &mut Store::<Fr>::default();
    //     let expected = s.read("354224848179261915075").unwrap();
    //     let terminal = s.get_cont_terminal();
    //     nova_test_full_aux(
    //         s,
    //         "(letrec ((next (lambda (a b n target)
    //                  (if (eq n target)
    //                      a
    //                      (next b
    //                          (+ a b)
    //                          (+ 1 n)
    //                         target))))
    //                 (fib (next 0 1 0)))
    //             (fib 100))",
    //         Some(expected),
    //         None,
    //         Some(terminal),
    //         None,
    //         4841,
    //         5,
    //         false,
    //     );
    // }

    #[test]
    fn outer_prove_terminal_continuation_regression() {
        let s = &mut Store::<Fr>::default();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((a (lambda (x) (cons 2 2))))
                     (a 1))",
            None,
            None,
            Some(terminal),
            None,
            9,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_chained_functional_commitment() {
        let s = &mut Store::<Fr>::default();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            "(letrec ((secret 12345)
                       (a (lambda (acc x)
                            (let ((acc (+ acc x)))
                              (cons acc (cons secret (a acc)))))))
                (a 0 5))",
            None,
            None,
            Some(terminal),
            None,
            39,
        );
    }

    #[test]
    fn outer_prove_begin_empty() {
        let s = &mut Store::<Fr>::default();
        let expected = s.nil();
        let terminal = s.get_cont_terminal();
        nova_test_aux(s, "(begin)", Some(expected), None, Some(terminal), None, 2);
    }

    #[test]
    fn outer_prove_begin_emit() {
        let s = &mut Store::<Fr>::default();
        let expr = "(begin (emit 1) (emit 2) (emit 3))";
        let expected_expr = s.num(3);
        let expected_emitted = vec![s.num(1), s.num(2), s.num(3)];
        nova_test_aux(
            s,
            expr,
            Some(expected_expr),
            None,
            None,
            Some(expected_emitted),
            13,
        );
    }

    #[test]
    fn outer_prove_str_car() {
        let s = &mut Store::<Fr>::default();
        let expected_a = s.read(r#"#\a"#).unwrap();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            r#"(car "apple")"#,
            Some(expected_a),
            None,
            Some(terminal),
            None,
            2,
        );
    }

    #[test]
    fn outer_prove_str_cdr() {
        let s = &mut Store::<Fr>::default();
        let expected_pple = s.read(r#" "pple" "#).unwrap();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            r#"(cdr "apple")"#,
            Some(expected_pple),
            None,
            Some(terminal),
            None,
            2,
        );
    }

    #[test]
    fn outer_prove_str_car_empty() {
        let s = &mut Store::<Fr>::default();
        let expected_nil = s.nil();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            r#"(car "")"#,
            Some(expected_nil),
            None,
            Some(terminal),
            None,
            2,
        );
    }

    #[test]
    fn outer_prove_str_cdr_empty() {
        let s = &mut Store::<Fr>::default();
        let expected_empty_str = s.intern_str(&"");
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            r#"(cdr "")"#,
            Some(expected_empty_str),
            None,
            Some(terminal),
            None,
            2,
        );
    }

    #[test]
    fn outer_prove_str_cons() {
        let s = &mut Store::<Fr>::default();
        let expected_apple = s.read(r#" "apple" "#).unwrap();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            r#"(cons #\a "pple")"#,
            Some(expected_apple),
            None,
            Some(terminal),
            None,
            3,
        );
    }

    #[test]
    fn outer_prove_one_arg_cons_error() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, r#"(cons "")"#, None, None, Some(error), None, 1);
    }

    #[test]
    fn outer_prove_car_nil() {
        let s = &mut Store::<Fr>::default();
        let expected = s.nil();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            r#"(car NIL)"#,
            Some(expected),
            None,
            Some(terminal),
            None,
            2,
        );
    }

    #[test]
    fn outer_prove_cdr_nil() {
        let s = &mut Store::<Fr>::default();
        let expected = s.nil();
        let terminal = s.get_cont_terminal();
        nova_test_aux(
            s,
            r#"(cdr NIL)"#,
            Some(expected),
            None,
            Some(terminal),
            None,
            2,
        );
    }

    #[test]
    fn outer_prove_car_cdr_invalid_tag_error_sym() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, r#"(car car)"#, None, None, Some(error), None, 2);
        nova_test_aux(s, r#"(cdr car)"#, None, None, Some(error), None, 2);
    }

    #[test]
    fn outer_prove_car_cdr_invalid_tag_error_char() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, r#"(car #\a)"#, None, None, Some(error), None, 2);
        nova_test_aux(s, r#"(cdr #\a)"#, None, None, Some(error), None, 2);
    }

    #[test]
    fn outer_prove_car_cdr_invalid_tag_error_num() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(s, r#"(car 42)"#, None, None, Some(error), None, 2);
        nova_test_aux(s, r#"(cdr 42)"#, None, None, Some(error), None, 2);
    }

    #[test]
    fn outer_prove_car_cdr_invalid_tag_error_lambda() {
        let s = &mut Store::<Fr>::default();
        let error = s.get_cont_error();
        nova_test_aux(
            s,
            r#"(car (lambda (x) x))"#,
            None,
            None,
            Some(error),
            None,
            2,
        );
        nova_test_aux(
            s,
            r#"(cdr (lambda (x) x))"#,
            None,
            None,
            Some(error),
            None,
            2,
        );
    }
}
