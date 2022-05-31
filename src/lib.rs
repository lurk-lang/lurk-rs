#![allow(clippy::single_match, clippy::type_complexity)]

extern crate core;

#[macro_use]
extern crate alloc;

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub mod circuit;
pub mod eval;
pub mod field;
pub mod parser;
pub mod proof;
pub mod repl;
pub mod scalar_store;
pub mod store;
#[cfg(feature = "wasm")]
pub mod wasm;
pub mod writer;

mod num;
pub use num::Num;

pub const TEST_SEED: [u8; 16] = [
    0x62, 0x59, 0x5d, 0xbe, 0x3d, 0x76, 0x3d, 0x8d, 0xdb, 0x17, 0x32, 0x37, 0x06, 0x54, 0xe5, 0xbc,
];

#[cfg(test)]
pub mod test {
    use quickcheck::Gen;
    use rand::Rng;

    // This is a useful testing utility for generating Arbitrary instances of
    // enums, by providing generators for each variant, plus a frequency weight
    // for how often to choose that variant. It's included in lib::test to make
    // it easier to import in the test modules of specific submodules.
    pub fn frequency<T, F: Fn(&mut Gen) -> T>(g: &mut Gen, gens: Vec<(i64, F)>) -> T {
        if gens.iter().any(|(v, _)| *v < 0) {
            panic!("Negative weight");
        }
        let sum: i64 = gens.iter().map(|x| x.0).sum();
        let mut rng = rand::thread_rng();
        let mut weight: i64 = rng.gen_range(1..=sum);
        // let mut weight: i64 = g.rng.gen_range(1, sum);
        for gen in gens {
            if weight - gen.0 <= 0 {
                return gen.1(g);
            } else {
                weight -= gen.0;
            }
        }
        panic!("Calculation error for weight = {}", weight);
    }
}
