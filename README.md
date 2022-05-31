# Lurk

[![CircleCI](https://circleci.com/gh/lurk-lang/lurk-rs.svg?style=shield)](https://circleci.com/gh/lurk-lang/lurk-rs)
![minimum rustc 1.60][msrv-image]
![crates.io][crates-image]

[msrv-image]: https://img.shields.io/badge/rustc-1.60+-blue.svg
[crates-image]: https://img.shields.io/crates/v/lurk.svg

# Disclaimer

**DISCLAIMER:** Lurk is an early research-stage language. Neither the cryptography nor the software has been audited, and there is currently no trusted setup for Groth16 circuits. Do not use Lurk in production environments or anywhere else that security is necessary.

# Overview

Lurk is a statically scoped dialect of Lisp, influenced by Scheme and Common Lisp. A language specification and reference implementation focused on describing and developing the core language can be found in the [`lurk`](https://github.com/lurk-lang/lurk) repo.

- [Lurk Language Specification](https://github.com/lurk-lang/lurk/blob/master/spec/v0-1.md)

Lurk's distinguishing feature relative to most programming languages is that correct execution of Lurk programs can be directly proved using zk-SNARKs. The resulting proofs are succinct: they are relatively small, can be verified quickly, and they reveal only the information explicitly contained in the statement to be proved.

Lurk's distinguishing feature relative to most zk-SNARK authoring languages is that Lurk is Turing complete, so arbitrary computational claims can be made and proved (subject to resource limitations, obviously). Because Lurk is a Lisp, its code is simply Lurk data, and any Lurk data can be directly evaluated as a Lurk program. Lurk constructs compound data using SNARK-friendly Poseidon hashes (provided by [Neptune](https://github.com/filecoin-project/neptune)), so its data is naturally content-addressable.

# Proofs

Integration with backend proving systems and tooling for proof generation are both still very early. Performance and user experience are poor, but simple examples can be found in the [fcomm example directory](fcomm/README.md).

# Backends
- The `fcomm` example uses Groth16/[SnarkPack](https://eprint.iacr.org/2021/529)[+](https://github.com/filecoin-project/bellperson/pull/257) to generate succinct (but somewhat large) proofs, using Bls12-381.
- The forthcoming Nova backend will use the [Nova proving system](https://github.com/microsoft/Nova) and the Pasta Curves.
- Future work may target Halo2 or other proving systems.

It is an explicit design goal that statements about the evaluation of Lurk programs have identical semantic meaning across backends, with the qualification that Lurk language instances are themselves parameterized on scalar field and hash function. When backends use the same scalar field and hash function, equivalent proofs can be generated across backends. This is because the concrete representation of content-addressed data is fixed.

# Performance

Lurk backend integration is still immature, so current performance is not representative. As a rough approximation, we estimate that for entirely general computation using Lurk's universal circuit, Nova proving throughput will be on the order of 1,000 iterations per second per GPU. We expect that most compute-heavy applications will use optimized 'coprocessor' circuits, which will  dramatically improve performance. Planned improvements to Nova will allow for smaller inner circuits, further improving throughput -- and for full parallelization of reduction proofs.

# (WIP) Specs
- [Circuit Spec](spec/main.pdf)
- [Evaluation Spec](spec/eval.md)
- [Reduction Notes](spec/reduction-notes.md)

---
# Build

## Submodules

Lurk source files used in tests are in the [lurk-lib](https://github.com/lurk-lang/lurk-lib) submodule. You must
initialize and update submodules before test will pass.

## Wasm Web Example

### Install prerequisites

- wasm32 target:
```
rustup target add wasm32-unknown-unknown
```
- wasm-pack:
```
cargo install wasm-pack
```
- wasm-ld linker:
```
# Ubuntu
sudo apt install llvm lld-14
# Mac
brew install llvm
# Add llvm to homebrew's PATH variable, e.g. one of the following
export PATH=/usr/local/opt/llvm/bin:$PATH
export PATH=/opt/homebrew/Cellar/llvm/13.0.1_1/bin:$PATH
# Verify installation
llc --version
```
- [clang](https://clang.llvm.org/get_started.html)
- [yarn](https://classic.yarnpkg.com/lang/en/docs/install/#mac-stable) or [npm](https://nodejs.org/en/download/package-manager/)
- [webpack](https://webpack.js.org/guides/installation/)

### Build with wasm-pack

* Linux
```
wasm-pack build --no-default-features --features wasm
```
* Mac
```
CC=clang AR=llvm-ar wasm-pack build --no-default-features --features wasm
```
### Build web evaluator

Building with `wasm-pack` will generate a `pkg` folder with the javascript wasm artifacts (Lurk wasm bindings). Once this is available, we can launch the web example.

Assuming we have `yarn` or `npm` installed, install `webpack` using the following command:

```
cd web

# Using yarn
yarn add -D webpack-cli 

# Using npm
npm install --save-dev webpack
```

After webpack installation, run the following yarn commands.
```
yarn install
yarn start
```

Go to [localhost:8080](http://localhost:8080) to view the evaluator


## Repl

```
cargo run --example repl
```

Or use the wrapper script:

```
bin/lurkrs
```

```
➜  lurk-rs ✗ bin/lurkrs
    Finished release [optimized] target(s) in 0.06s
     Running `target/release/examples/repl`
Lurk REPL welcomes you.
> (let ((square (lambda (x) (* x x)))) (square 8))
[9 iterations] => 64
>
```

Or enable `info` log-level for a trace of reduction frames:
```
➜  lurk-rs ✗ RUST_LOG=info bin/lurkrs
    Finished release [optimized] target(s) in 0.05s
     Running `target/release/examples/repl`
Lurk REPL welcomes you.
> (let ((square (lambda (x) (* x x)))) (square 8))
INFO  lurk::eval > Frame: 0
        Expr: (LET ((SQUARE (LAMBDA (X) (* X X)))) (SQUARE 8))
        Env: NIL
        Cont: Outermost
INFO  lurk::eval > Frame: 1
        Expr: (LAMBDA (X) (* X X))
        Env: NIL
        Cont: Let{ var: SQUARE, body: (SQUARE 8), saved_env: NIL, continuation: Outermost }
INFO  lurk::eval > Frame: 2
        Expr: (SQUARE 8)
        Env: ((SQUARE . <FUNCTION (X) . ((* X X))>))
        Cont: Tail{ saved_env: NIL, continuation: Outermost }
INFO  lurk::eval > Frame: 3
        Expr: SQUARE
        Env: ((SQUARE . <FUNCTION (X) . ((* X X))>))
        Cont: Call{ unevaled_arg: 8, saved_env: ((SQUARE . <FUNCTION (X) . ((* X X))>)), continuation: Tail{ saved_env: NIL, continuation: Outermost } }
INFO  lurk::eval > Frame: 4
        Expr: 8
        Env: ((SQUARE . <FUNCTION (X) . ((* X X))>))
        Cont: Call2{ function: <FUNCTION (X) . ((* X X))>, saved_env: ((SQUARE . <FUNCTION (X) . ((* X X))>)), continuation: Tail{ saved_env: NIL, continuation: Outermost } }
INFO  lurk::eval > Frame: 5
        Expr: (* X X)
        Env: ((X . 8))
        Cont: Tail{ saved_env: NIL, continuation: Outermost }
INFO  lurk::eval > Frame: 6
        Expr: X
        Env: ((X . 8))
        Cont: Binop{ operator: Product, unevaled_args: (X), saved_env: ((X . 8)), continuation: Tail{ saved_env: NIL, continuation: Outermost } }
INFO  lurk::eval > Frame: 7
        Expr: X
        Env: ((X . 8))
        Cont: Binop2{ operator: Product, evaled_arg: 8, continuation: Tail{ saved_env: NIL, continuation: Outermost } }
INFO  lurk::eval > Frame: 8
        Expr: Thunk{ value: 64 => cont: Outermost}
        Env: NIL
        Cont: Dummy
INFO  lurk::eval > Frame: 9
        Expr: 64
        Env: NIL
        Cont: Terminal
[9 iterations] => 64
> 
```

## Nix

You can enter into a [Nix](https://nixos.org) shell with the appropriate
dependencies for Lurk with
```
$ nix-shell
```

And then building with Cargo as usual:

```
$ cargo build
```

## Troubleshooting

- Wasm build: `Error: failed to build archive: section too large`
Run `cargo clean`, then try compiling with `CC=clang AR=llvm-ar`

## License

MIT or Apache 2.0
