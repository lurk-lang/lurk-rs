use std::io;

use crate::field::LurkField;
use crate::store::{ContPtr, Continuation, Expression, Ptr, Store};

pub trait Write<F: LurkField> {
    fn fmt<W: io::Write>(&self, store: &Store<F>, w: &mut W) -> io::Result<()>;
    fn fmt_to_string(&self, store: &Store<F>) -> String {
        let mut out = Vec::new();
        self.fmt(store, &mut out).expect("preallocated");
        String::from_utf8(out).expect("I know it")
    }
}

impl<F: LurkField> Write<F> for Ptr<F> {
    fn fmt<W: io::Write>(&self, store: &Store<F>, w: &mut W) -> io::Result<()> {
        use crate::store::{Pointer, ScalarPointer};
        if self.is_opaque() {
            // This should never fail.
            write!(w, "<Opaque ")?;
            write!(w, "{:?}", self.tag())?;

            if let Some(x) = store.get_expr_hash(self) {
                write!(w, " ")?;
                crate::store::Expression::Num(crate::num::Num::Scalar(*x.value())).fmt(store, w)?;
            }
            write!(w, ">")
        } else if let Some(expr) = store.fetch(self) {
            expr.fmt(store, w)
        } else {
            Ok(())
        }
    }
}

impl<F: LurkField> Write<F> for ContPtr<F> {
    fn fmt<W: io::Write>(&self, store: &Store<F>, w: &mut W) -> io::Result<()> {
        if let Some(cont) = store.fetch_cont(self) {
            cont.fmt(store, w)
        } else {
            Ok(())
        }
    }
}

impl<F: LurkField> Write<F> for Expression<'_, F> {
    fn fmt<W: io::Write>(&self, store: &Store<F>, w: &mut W) -> io::Result<()> {
        use Expression::*;

        match self {
            Nil => write!(w, "NIL"),
            Sym(s) => write!(w, "{}", s),
            Str(s) => write!(w, "\"{}\"", s),
            Fun(arg, body, _closed_env) => {
                let is_zero_arg = *arg == store.get_sym("_", true).expect("dummy_arg (_) missing");
                let arg = store.fetch(arg).unwrap();
                let body = store.fetch(body).unwrap();
                write!(w, "<FUNCTION (")?;
                if !is_zero_arg {
                    arg.fmt(store, w)?;
                }
                write!(w, ") ")?;
                assert!(body.is_cons(), "Fun body should be a non-empty list.");
                body.print_tail(store, w)?;
                write!(w, ">")
            }
            Num(n) => write!(w, "{}", n),
            Thunk(f) => {
                write!(w, "Thunk{{ value: ")?;
                f.value.fmt(store, w)?;
                write!(w, " => cont: ")?;
                f.continuation.fmt(store, w)?;
                write!(w, "}}")
            }
            Cons(_, _) => {
                write!(w, "(")?;
                self.print_tail(store, w)
            }
            Comm(secret, payload) => {
                // This requires a run-time coercion.
                // Consider implementing the equivalent of CL's #. reader macro to let this happen at read-time.
                write!(w, "(comm ")?;
                let c = store.commitment_hash(*secret, store.get_expr_hash(payload).unwrap());
                Num(crate::num::Num::Scalar(c)).fmt(store, w)?;
                write!(w, ")")
            }
            Opaque(f) => f.fmt(store, w),
            Char(c) => {
                write!(w, "#\\{}", c)
            }
        }
    }
}

impl<F: LurkField> Expression<'_, F> {
    fn print_tail<W: io::Write>(&self, store: &Store<F>, w: &mut W) -> io::Result<()> {
        match self {
            Expression::Nil => write!(w, ")"),
            Expression::Cons(car, cdr) => {
                let car = store.fetch(car);
                let cdr = store.fetch(cdr);
                let fmt_car = |store, w: &mut W| {
                    if let Some(car) = car {
                        car.fmt(store, w)
                    } else {
                        write!(w, "<Opaque>")
                    }
                };
                let fmt_cdr = |store, w: &mut W| {
                    if let Some(cdr) = cdr {
                        cdr.fmt(store, w)
                    } else {
                        write!(w, "<Opaque>")
                    }
                };

                match cdr {
                    Some(Expression::Nil) => {
                        fmt_car(store, w)?;
                        write!(w, ")")
                    }
                    Some(Expression::Cons(_, _)) => {
                        fmt_car(store, w)?;
                        write!(w, " ")?;
                        if let Some(cdr) = cdr {
                            cdr.print_tail(store, w)
                        } else {
                            write!(w, "<Opaque Tail>")
                        }
                    }
                    Some(_) => {
                        fmt_car(store, w)?;
                        write!(w, " . ")?;
                        fmt_cdr(store, w)?;
                        write!(w, ")")
                    }
                    None => write!(w, "<Opaque>"),
                }
            }
            _ => {
                unreachable!()
            }
        }
    }
}

impl<F: LurkField> Write<F> for Continuation<F> {
    fn fmt<W: io::Write>(&self, store: &Store<F>, w: &mut W) -> io::Result<()> {
        match self {
            Continuation::Outermost => write!(w, "Outermost"),
            Continuation::Call0 { continuation } => {
                write!(w, "Call0{{ continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Call {
                unevaled_arg,
                saved_env,
                continuation,
            } => {
                write!(w, "Call{{ unevaled_arg: ")?;
                unevaled_arg.fmt(store, w)?;
                write!(w, ", saved_env: ")?;
                saved_env.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Call2 {
                function,
                saved_env,
                continuation,
            } => {
                write!(w, "Call2{{ function: ")?;
                function.fmt(store, w)?;
                write!(w, ", saved_env: ")?;
                saved_env.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Tail {
                saved_env,
                continuation,
            } => {
                write!(w, "Tail{{ saved_env: ")?;
                saved_env.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Error => write!(w, "Error"),
            Continuation::Lookup {
                saved_env,
                continuation,
            } => {
                write!(w, "Lookup{{ saved_env: ")?;
                saved_env.fmt(store, w)?;
                write!(w, ", ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Unop {
                operator,
                continuation,
            } => {
                write!(w, "Unop{{ operator: {}, continuation: ", operator)?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Binop {
                operator,
                saved_env,
                unevaled_args,
                continuation,
            } => {
                write!(w, "Binop{{ operator: ")?;
                write!(w, "{}, unevaled_args: ", operator)?;
                unevaled_args.fmt(store, w)?;
                write!(w, ", saved_env: ")?;
                saved_env.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Binop2 {
                operator,
                evaled_arg,
                continuation,
            } => {
                write!(w, "Binop2{{ operator: {}, evaled_arg: ", operator)?;
                evaled_arg.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Relop {
                operator,
                saved_env,
                unevaled_args,
                continuation,
            } => {
                write!(w, "Relop{{ operator: {}, saved_env: ", operator)?;
                saved_env.fmt(store, w)?;
                write!(w, ", unevaled_args: ")?;
                unevaled_args.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, ")")
            }
            Continuation::Relop2 {
                operator,
                evaled_arg,
                continuation,
            } => {
                write!(w, "Relop2{{ operator: {}, evaled_ag: ", operator)?;
                evaled_arg.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::If {
                unevaled_args,
                continuation,
            } => {
                write!(w, "If{{ unevaled_args: ")?;
                unevaled_args.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Let {
                var,
                body,
                saved_env,
                continuation,
            } => {
                write!(w, "Let{{ var: ")?;
                var.fmt(store, w)?;
                write!(w, ", body: ")?;
                body.fmt(store, w)?;
                write!(w, ", saved_env: ")?;
                saved_env.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::LetRec {
                var,
                saved_env,
                body,
                continuation,
            } => {
                write!(w, "LetRec{{var: ")?;
                var.fmt(store, w)?;
                write!(w, ", saved_env: ")?;
                saved_env.fmt(store, w)?;
                write!(w, ", body: ")?;
                body.fmt(store, w)?;
                write!(w, ", continuation: ")?;
                continuation.fmt(store, w)?;
                write!(w, " }}")
            }
            Continuation::Dummy => write!(w, "Dummy"),
            Continuation::Terminal => write!(w, "Terminal"),
            Continuation::Emit {
                continuation: _continuation,
            } => {
                write!(w, "Emit")?;
                write!(w, "<CONTINUATION>") // Omit continuation for clarity when logging and using output.
                                            // write!(w, " {{ continuation: ")?;
                                            // continuation.fmt(store, w)?;
                                            // write!(w, " }}")
            }
        }
    }
}
