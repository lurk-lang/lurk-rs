use blstrs::Scalar as Fr;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lurk::{
    eval::{empty_sym_env, Evaluator},
    field::LurkField,
    store::{Ptr, Store},
};

fn go_base<F: LurkField>(store: &mut Store<F>, a: u64, b: u64) -> Ptr<F> {
    let program = format!(
        r#"
(let ((foo (lambda (a b)
              (letrec ((aux (lambda (i a x)
                               (if (= i b)
                                     x
                                     (let ((x (+ x a))
                                            (a (+ a (* b 2))))
                                       (aux (+ i 1) a x))))))
                       (let ((x (+ (* a b) 4)))
                         (aux 0 a x))))))
  (foo {} {}))
"#,
        a, b
    );

    store.read(&program).unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    let limit = 1_000_000_000;

    c.bench_function("go_base_10_16_bls12", |b| {
        let mut store = Store::default();
        let ptr = go_base::<Fr>(&mut store, black_box(10), black_box(16));

        b.iter(|| {
            let result = Evaluator::new(ptr, empty_sym_env(&store), &mut store, limit).eval();
            black_box(result)
        })
    });

    c.bench_function("go_base_10_160_bls12", |b| {
        let mut store = Store::default();
        let ptr = go_base::<Fr>(&mut store, black_box(10), black_box(160));

        b.iter(|| {
            let result = Evaluator::new(ptr, empty_sym_env(&store), &mut store, limit).eval();
            black_box(result)
        })
    });

    c.bench_function("go_base_10_16_pasta_pallas", |b| {
        let mut store = Store::default();
        let ptr = go_base::<pasta_curves::Fp>(&mut store, black_box(10), black_box(16));

        b.iter(|| {
            let result = Evaluator::new(ptr, empty_sym_env(&store), &mut store, limit).eval();
            black_box(result)
        })
    });

    c.bench_function("go_base_10_160_pasta_pallas", |b| {
        let mut store = Store::default();
        let ptr = go_base::<pasta_curves::Fp>(&mut store, black_box(10), black_box(160));

        b.iter(|| {
            let result = Evaluator::new(ptr, empty_sym_env(&store), &mut store, limit).eval();
            black_box(result)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
