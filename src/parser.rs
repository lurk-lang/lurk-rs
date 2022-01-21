use std::iter::Peekable;

use crate::pool::{Pool, Ptr};

impl Pool {
    pub fn read(&mut self, input: &str) -> Option<Ptr> {
        let mut chars = input.chars().peekable();

        self.read_next(&mut chars)
    }

    // For now, this is only used for REPL/CLI commands.
    pub fn read_string<T: Iterator<Item = char>>(
        &mut self,
        chars: &mut Peekable<T>,
    ) -> Option<Ptr> {
        let mut result = String::new();

        if let Some('"') = skip_whitespace_and_peek(chars) {
            chars.next();
            while let Some(&c) = chars.peek() {
                chars.next();
                // TODO: This does not handle any escaping, so strings containing " cannot be read.
                if c == '"' {
                    let str = self.alloc_str(result);
                    return Some(str);
                } else {
                    result.push(c);
                }
            }
            None
        } else {
            None
        }
    }

    pub fn read_maybe_meta<T: Iterator<Item = char>>(
        &mut self,
        chars: &mut Peekable<T>,
    ) -> Option<(Ptr, bool)> {
        if let Some(c) = skip_whitespace_and_peek(chars) {
            match c {
                '!' => {
                    chars.next();
                    if let Some(s) = self.read_string(chars) {
                        Some((s, true))
                    } else if let Some((e, is_meta)) = self.read_maybe_meta(chars) {
                        assert!(!is_meta);
                        Some((e, true))
                    } else {
                        None
                    }
                }
                _ => self.read_next(chars).map(|expr| (expr, false)),
            }
        } else {
            None
        }
    }

    pub fn read_next<T: Iterator<Item = char>>(&mut self, chars: &mut Peekable<T>) -> Option<Ptr> {
        while let Some(&c) = chars.peek() {
            if let Some(next_expr) = match c {
                '(' => self.read_list(chars),
                '0'..='9' => self.read_number(chars),
                ' ' | '\t' | '\n' | '\r' => {
                    // Skip whitespace.
                    chars.next();
                    continue;
                }
                '\'' => {
                    chars.next();
                    let quote = self.alloc_sym("quote");
                    let quoted = self.read_next(chars)?;
                    let inner = self.alloc_list(&[quoted]);
                    Some(self.alloc_cons(quote, inner))
                }
                '\"' => self.read_string(chars),
                ';' => {
                    chars.next();
                    if skip_line_comment(chars) {
                        continue;
                    } else {
                        None
                    }
                }
                x if is_symbol_char(&x, true) => self.read_symbol(chars),
                _ => {
                    panic!("bad input character: {}", c);
                }
            } {
                return Some(next_expr);
            }
        }
        None
    }

    // In this context, 'list' includes improper lists, i.e. dotted cons-pairs like (1 . 2).
    fn read_list<T: Iterator<Item = char>>(&mut self, chars: &mut Peekable<T>) -> Option<Ptr> {
        if let Some(&c) = chars.peek() {
            match c {
                '(' => {
                    chars.next(); // Discard.
                    self.read_tail(chars)
                }
                _ => None,
            }
        } else {
            None
        }
    }

    // Read the tail of a list.
    fn read_tail<T: Iterator<Item = char>>(&mut self, chars: &mut Peekable<T>) -> Option<Ptr> {
        if let Some(c) = skip_whitespace_and_peek(chars) {
            match c {
                ')' => {
                    chars.next();
                    Some(self.alloc_nil())
                }
                '.' => {
                    chars.next();
                    let cdr = self.read_next(chars).unwrap();
                    let remaining_tail = self.read_tail(chars).unwrap();
                    assert!(remaining_tail.is_nil());

                    Some(cdr)
                }
                _ => {
                    let car = self.read_next(chars).unwrap();
                    let rest = self.read_tail(chars).unwrap();
                    Some(self.alloc_cons(car, rest))
                }
            }
        } else {
            panic!("premature end of input");
        }
    }

    fn read_number<T: Iterator<Item = char>>(&mut self, chars: &mut Peekable<T>) -> Option<Ptr> {
        // As written, read_number assumes the next char is known to be a digit.
        // So it will never return None.
        let mut acc = 0;
        let ten = 10;

        while let Some(&c) = chars.peek() {
            if is_digit_char(&c) {
                if acc != 0 {
                    acc *= ten;
                }
                let digit_char = chars.next().unwrap();
                let digit = digit_char.to_digit(10).unwrap();
                let n: u64 = digit.into();
                acc += n;
            } else {
                break;
            }
        }
        Some(self.alloc_num(acc))
    }

    fn read_symbol<T: Iterator<Item = char>>(&mut self, chars: &mut Peekable<T>) -> Option<Ptr> {
        let mut name = String::new();
        let mut is_initial = true;
        while let Some(&c) = chars.peek() {
            if is_symbol_char(&c, is_initial) {
                let c = chars.next().unwrap();
                name.push(c);
            } else {
                break;
            }
            is_initial = false;
        }

        Some(self.alloc_sym(name))
    }
}

fn is_symbol_char(c: &char, initial: bool) -> bool {
    match c {
        // FIXME: suppport more than just alpha.
        'a'..='z' | 'A'..='Z' | '+' | '-' | '*' | '/' | '=' | ':' => true,
        _ => {
            if initial {
                false
            } else {
                matches!(c, '0'..='9')
            }
        }
    }
}

fn is_digit_char(c: &char) -> bool {
    matches!(c, '0'..='9')
}

#[allow(dead_code)]
fn is_reserved_char(c: &char) -> bool {
    matches!(c, '(' | ')' | '.')
}

fn is_whitespace_char(c: &char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

fn is_comment_char(c: &char) -> bool {
    matches!(c, ';')
}

fn is_line_end_char(c: &char) -> bool {
    matches!(c, '\n' | '\r')
}

// Skips whitespace and comments, returning the next character, if any.
fn skip_whitespace_and_peek<T: Iterator<Item = char>>(chars: &mut Peekable<T>) -> Option<char> {
    while let Some(&c) = chars.peek() {
        if is_whitespace_char(&c) {
            chars.next();
        } else if is_comment_char(&c) {
            skip_line_comment(chars);
        } else {
            return Some(c);
        }
    }
    None
}

// Returns true if comment ends with a line end character.
// If false, this comment is unterminated and is the end of input.
fn skip_line_comment<T: Iterator<Item = char>>(chars: &mut Peekable<T>) -> bool {
    while let Some(&c) = chars.peek() {
        if !is_line_end_char(&c) {
            chars.next();
        } else {
            return true;
        }
    }
    false

    //chars.skip_while(|c| *c != '\n' && *c != '\r');
    //     }
    // };
}

#[cfg(test)]
mod test {
    use crate::writer::Write;

    use super::*;

    #[test]
    fn read_sym() {
        let test = |input, expected: &str| {
            let mut pool = Pool::default();
            let ptr = &pool.read(input).unwrap();
            let expr = pool.fetch(&ptr).unwrap();

            assert_eq!(expected, expr.as_sym_str().unwrap());
        };

        test("asdf", "ASDF");
        test("asdf ", "ASDF");
        test("asdf(", "ASDF");
        test(" asdf", "ASDF");
        test(" asdf ", "ASDF");
        test(
            "
asdf(", "ASDF",
        );
    }

    #[test]
    fn read_nil() {
        let mut pool = Pool::default();
        let expr = pool.read("nil").unwrap();
        assert!(expr.is_nil());
    }

    #[test]
    fn read_num() {
        let test = |input, expected: u64| {
            let mut pool = Pool::default();
            let expr = pool.read(input).unwrap();
            assert_eq!(pool.alloc_num(expected), expr);
        };
        test("123", 123);
        test("0987654321", 987654321);
        test("123)", 123);
        test("123 ", 123);
        test("123z", 123);
        test(" 123", 123);
        test(
            "
0987654321",
            987654321,
        );
    }

    #[test]
    fn read_list() {
        let mut pool = Pool::default();
        let test = |pool: &mut Pool, input, expected| {
            let expr = pool.read(input).unwrap();
            assert_eq!(expected, &expr);
        };

        let a = pool.alloc_num(123);
        let b = pool.alloc_nil();
        let expected = pool.alloc_cons(a, b);
        test(&mut pool, "(123)", &expected);

        let a = pool.alloc_num(321);
        let expected2 = pool.alloc_cons(a, expected);
        test(&mut pool, "(321 123)", &expected2);

        let a = pool.alloc_sym("PUMPKIN");
        let expected3 = pool.alloc_cons(a, expected2);
        test(&mut pool, "(pumpkin 321 123)", &expected3);

        let expected4 = pool.alloc_cons(expected, pool.alloc_nil());
        test(&mut pool, "((123))", &expected4);

        let (a, b) = (pool.alloc_num(321), pool.alloc_nil());
        let alt = pool.alloc_cons(a, b);
        let expected5 = pool.alloc_cons(alt, expected4);
        test(&mut pool, "((321) (123))", &expected5);

        let expected6 = pool.alloc_cons(expected2, expected3);
        test(&mut pool, "((321 123) pumpkin 321 123)", &expected6);

        let (a, b) = (pool.alloc_num(1), pool.alloc_num(2));
        let pair = pool.alloc_cons(a, b);
        let list = [pair, pool.alloc_num(3)];
        let expected7 = pool.alloc_list(&list);
        test(&mut pool, "((1 . 2) 3)", &expected7);
    }

    #[test]
    fn read_improper_list() {
        let mut pool = Pool::default();
        let test = |pool: &mut Pool, input, expected| {
            let expr = pool.read(input).unwrap();
            assert_eq!(expected, &expr);
        };

        let (a, b) = (pool.alloc_num(123), pool.alloc_num(321));
        let expected = pool.alloc_cons(a, b);
        test(&mut pool, "(123 . 321)", &expected);

        assert_eq!(pool.read("(123 321)"), pool.read("(123 . ( 321 ))"))
    }
    #[test]
    fn read_print_expr() {
        let mut pool = Pool::default();
        let test = |pool: &mut Pool, input| {
            let expr = pool.read(input).unwrap();
            let output = expr.fmt_to_string(pool);
            assert_eq!(input, output);
        };

        test(&mut pool, "A");
        test(&mut pool, "(A . B)");
        test(&mut pool, "(A B C)");
        test(&mut pool, "(A (B) C)");
        test(&mut pool, "(A (B . C) (D E (F)) G)");
        // test(&mut pool, "'A");
        // test(&mut pool, "'(A B)");
    }

    #[test]
    fn read_maybe_meta() {
        let mut pool = Pool::default();
        let test = |pool: &mut Pool, input: &str, expected_ptr: Ptr, expected_meta: bool| {
            let mut chars = input.chars().peekable();

            match pool.read_maybe_meta(&mut chars).unwrap() {
                (ptr, meta) => {
                    assert_eq!(expected_ptr, ptr);
                    assert_eq!(expected_meta, meta);
                }
            };
        };

        let num = pool.alloc_num(123);
        test(&mut pool, "123", num, false);

        {
            let list = [pool.alloc_num(123), pool.alloc_num(321)];
            let l = pool.alloc_list(&list);
            test(&mut pool, " (123 321)", l, false);
        }
        {
            let list = [pool.alloc_num(123), pool.alloc_num(321)];
            let l = pool.alloc_list(&list);
            test(&mut pool, " !(123 321)", l, true);
        }
        {
            let list = [pool.alloc_num(123), pool.alloc_num(321)];
            let l = pool.alloc_list(&list);
            test(&mut pool, " ! (123 321)", l, true);
        }
        {
            let s = pool.alloc_sym("asdf");
            test(&mut pool, "!asdf", s, true);
        }
        {
            let s = pool.alloc_sym(":assert");
            let l = pool.alloc_list(&[s]);
            test(&mut pool, "!(:assert)", l, true);
        }
        {
            let s = pool.alloc_sym("asdf");
            test(
                &mut pool,
                ";; comment
!asdf",
                s,
                true,
            );
        }
    }
    #[test]
    fn is_keyword() {
        let mut pool = Pool::default();
        let kw = pool.alloc_sym(":UIOP");
        let not_kw = pool.alloc_sym("UIOP");

        assert!(pool.fetch(&kw).unwrap().is_keyword_sym());
        assert!(!pool.fetch(&not_kw).unwrap().is_keyword_sym());
    }

    #[test]
    fn read_string() {
        let mut pool = Pool::default();
        let test = |pool: &mut Pool, input: &str, expected: Option<Ptr>| {
            let maybe_string = pool.read_string(&mut input.chars().peekable());
            assert_eq!(expected, maybe_string);
        };

        let s = pool.alloc_str("asdf");
        test(&mut pool, "\"asdf\"", Some(s));
        test(&mut pool, "\"asdf", None);
        test(&mut pool, "asdf", None);
    }
    #[test]
    fn read_with_comments() {
        let mut pool = Pool::default();

        let test = |pool: &mut Pool, input: &str, expected: Option<Ptr>| {
            let res = pool.read(input);
            assert_eq!(expected, res);
        };

        let num = pool.alloc_num(321);
        test(
            &mut pool,
            ";123
321",
            Some(num),
        );
    }
}