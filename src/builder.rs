use std::cmp::Ordering;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use hashconsing::hash_coll::HConMap;
use hashconsing::HConsed;
use hashconsing::{HashConsign, HConsign};

use fugue::bytes::{ByteCast, Order};

use parking_lot::RwLock;
use thiserror::Error;

use z3::Config as Z3Config;
use z3::Context as Z3Context;
use z3::Solver as Z3Solver;

use z3::ast::Ast as Z3Ast;
use z3::ast::Bool as Z3Bool;
use z3::ast::BV as Z3BV;

use crate::path::PathManager;
use crate::value::ConcolicByteArray;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsatisfiable constraints")]
    UnSat
}

// We represent symbolic memory and registers as
// a mapping of symbolic expressions to offsets in either memory,
// registers, or temporary space.

pub type SymExpr = HConsed<Expr>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    name: String,
    bits: u32,
}

impl Var {
    pub fn get_name(&self) -> &str{
        return &self.name;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr {
    Bool(bool),
    Integer {
        value: u64,
        bits: u32,
    },
    Variable(Var),
    InternalVar(u64, u32),

    Not(SymExpr),
    And(SymExpr, SymExpr),
    Or(SymExpr, SymExpr),
    Xor(SymExpr, SymExpr),

    Neg(SymExpr),
    Add(SymExpr, SymExpr),
    Carry(SymExpr, SymExpr),
    SCarry(SymExpr, SymExpr),
    Sub(SymExpr, SymExpr),
    SBorrow(SymExpr, SymExpr),
    Mul(SymExpr, SymExpr),
    Div(SymExpr, SymExpr),
    SDiv(SymExpr, SymExpr),
    Rem(SymExpr, SymExpr),
    SRem(SymExpr, SymExpr),
    Shl(SymExpr, SymExpr),
    Shr(SymExpr, SymExpr),
    Sar(SymExpr, SymExpr),

    Less(SymExpr, SymExpr),
    SLess(SymExpr, SymExpr),
    LessEq(SymExpr, SymExpr),
    SLessEq(SymExpr, SymExpr),
    Greater(SymExpr, SymExpr),
    SGreater(SymExpr, SymExpr),
    GreaterEq(SymExpr, SymExpr),
    SGreaterEq(SymExpr, SymExpr),
    Equal(SymExpr, SymExpr),
    NotEqual(SymExpr, SymExpr),

    Concat(SymExpr, SymExpr),
    Extract { expr: SymExpr, lsb: usize, bits: usize },
    SignExtend { expr: SymExpr, bits: u32 },
    ZeroExtend { expr: SymExpr, bits: u32 },

    // `Abstracted` is intended to model symbolic relationships over black-boxes
    // as proposed by Vanhoef et al. in "Symbolic Execution of Security Protocol
    // Implementations: Handling Cryptographic Primitives".
    Relation { operator: &'static str, arguments: Vec<SymExpr>, result: SymExpr },
}

impl From<bool> for Expr {
    fn from(t: bool) -> Expr {
        Expr::Bool(t)
    }
}

impl From<u8> for Expr {
    fn from(t: u8) -> Expr {
        Expr::Integer { value: t as u64, bits: 8 }
    }
}

impl From<u16> for Expr {
    fn from(t: u16) -> Expr {
        Expr::Integer { value: t as u64, bits: 16 }
    }
}

impl From<u32> for Expr {
    fn from(t: u32) -> Expr {
        Expr::Integer { value: t as u64, bits: 32 }
    }
}

impl From<u64> for Expr {
    fn from(t: u64) -> Expr {
        Expr::Integer { value: t as u64, bits: 64 }
    }
}

impl From<i8> for Expr {
    fn from(t: i8) -> Expr {
        Expr::Integer { value: t as u8 as u64, bits: 8 }
    }
}

impl From<i16> for Expr {
    fn from(t: i16) -> Expr {
        Expr::Integer { value: t as u16 as u64, bits: 16 }
    }
}

impl From<i32> for Expr {
    fn from(t: i32) -> Expr {
        Expr::Integer { value: t as u32 as u64, bits: 32 }
    }
}

impl From<i64> for Expr {
    fn from(t: i64) -> Expr {
        Expr::Integer { value: t as u64, bits: 64 }
    }
}

#[derive(Clone)]
pub struct ExprBuilder(Arc<RwLock<ExprBuilderImpl>>);

impl ExprBuilder {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(ExprBuilderImpl::new())))
    }

    pub fn fork(&self) -> Self {
        Self(Arc::new(RwLock::new(self.0.read().fork())))
    }

    pub fn snapshot(&self) -> Self {
        Self(Arc::new(RwLock::new(self.0.read().snapshot())))
    }

    pub fn primitive_bytes<const N: usize>(&self) -> ConcolicByteArray<N> {
        self.0.write().primitive_bytes::<{ N }>()
    }

    pub fn zero(&self, bits: u32) -> SymExpr {
        self.0.write().zero(bits)
    }

    pub fn one(&self, bits: u32) -> SymExpr {
        self.0.write().one(bits)
    }

    pub fn constant<T: Into<Expr>>(&self, t: T) -> SymExpr {
        self.0.write().constant(t)
    }

    pub fn constant_with_size(&self, value: u64, bits: u32) -> SymExpr {
        self.0.write().constant_with_size(value, bits)
    }

    pub fn collect(&self) {
        self.0.write().collect()
    }

    pub fn variable<S: AsRef<str>>(&self, name: S, bits: u32) -> (Var, SymExpr) {
        self.0.write().variable(name, bits)
    }

    pub fn unnamed_variable(&self, bits: u32) -> (u64, SymExpr) {
        self.0.write().unnamed_variable(bits)
    }

    // Size in bits
    pub fn size(&self, expr: &SymExpr) -> usize {
        self.0.write().size(expr)
    }

    pub fn concat(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().concat(lhs, rhs)
    }

    pub fn extract(&self, expr: SymExpr, lsb: usize, bits: usize) -> SymExpr {
        self.0.write().extract(expr, lsb, bits)
    }

    // Extend by X bits e.g extend 8 bits to 32 bits -> sign_extend(expr, 24)
    pub fn sign_extend(&self, expr: SymExpr, by_bits: u32) -> SymExpr {
        self.0.write().sign_extend(expr, by_bits)
    }

    pub fn zero_extend(&self, expr: SymExpr, by_bits: u32) -> SymExpr {
        self.0.write().zero_extend(expr, by_bits)
    }

    // Extend to X bits e.g extend 8 bits to 32 bits -> sign_extend(expr, 32)
    pub fn sign_extend_to(&self, expr: SymExpr, to_bits: u32) -> SymExpr {
        self.0.write().sign_extend_to(expr, to_bits)
    }

    pub fn zero_extend_to(&self, expr: SymExpr, to_bits: u32) -> SymExpr {
        self.0.write().zero_extend_to(expr, to_bits)
    }

    pub fn not(&self, expr: SymExpr) -> SymExpr {
        self.0.write().not(expr)
    }

    pub fn and(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().and(lhs, rhs)
    }

    pub fn or(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().or(lhs, rhs)
    }

    pub fn xor(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().xor(lhs, rhs)
    }

    pub fn neg(&self, expr: SymExpr) -> SymExpr {
        self.0.write().neg(expr)
    }

    pub fn add(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().add(lhs, rhs)
    }

    pub fn carry(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().carry(lhs, rhs)
    }

    pub fn scarry(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().scarry(lhs, rhs)
    }

    pub fn sub(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().sub(lhs, rhs)
    }

    pub fn sborrow(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().sborrow(lhs, rhs)
    }

    pub fn mul(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().mul(lhs, rhs)
    }

    pub fn div(&self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().div(lhs, rhs)
    }

    pub fn signed_div(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().signed_div(lhs, rhs)
    }

    pub fn rem(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().rem(lhs, rhs)
    }

    pub fn signed_rem(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().signed_rem(lhs, rhs)
    }

    pub fn shift_left(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().shift_left(lhs, rhs)
    }

    pub fn shift_right(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().shift_right(lhs, rhs)
    }

    pub fn signed_shift_right(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().signed_shift_right(lhs, rhs)
    }

    pub fn equal(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().equal(lhs, rhs)
    }

    pub fn not_equal(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().not_equal(lhs, rhs)
    }

    pub fn less(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().less(lhs, rhs)
    }

    pub fn signed_less(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().signed_less(lhs, rhs)
    }

    pub fn less_eq(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().less_eq(lhs, rhs)
    }

    pub fn signed_less_eq(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().signed_less_eq(lhs, rhs)
    }

    pub fn greater(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().greater(lhs, rhs)
    }

    pub fn signed_greater(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().signed_greater(lhs, rhs)
    }

    pub fn greater_eq(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().signed_greater(lhs, rhs)
    }

    pub fn signed_greater_eq(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.0.write().signed_greater_eq(lhs, rhs)
    }

    /*
    pub fn solve_value<T: ByteCast, O: Order>(&self, expr: SymExpr, path_manager: &PathManager) -> Result<T, Error> {
        self.0.write().solve_value::<T, O>(expr, path_manager)
    }

    pub fn solve(&self, expr: &SymExpr, vars: Vec<SymExpr>, path_manager: &PathManager) -> Option<HashMap<SymExpr, Option<u64>>> {
        self.0.write().solve(expr, vars, path_manager)
    }
    */
}

// We use hash consing to construct expression trees
pub struct ExprBuilderImpl {
    consign: HConsign<Expr>,
    size_cache: HashMap<u64, usize>,
    ivar_counter: u64,
}

impl Clone for ExprBuilderImpl {
    fn clone(&self) -> Self {
        Self {
            consign: self.consign.unsafe_clone(),
            size_cache: self.size_cache.clone(),
            ivar_counter: self.ivar_counter,
        }
    }
}

impl ExprBuilderImpl {
    pub fn new() -> Self {
        Self {
            consign: HConsign::empty(),
            size_cache: HashMap::new(),
            ivar_counter: 0,
        }
    }

    #[inline]
    pub fn fork(&self) -> Self {
        Self {
            consign: self.consign.unsafe_clone(),
            size_cache: HashMap::new(),
            ivar_counter: self.ivar_counter,
        }
    }

    #[inline]
    pub fn snapshot(&self) -> Self {
        self.clone()
    }

    pub fn primitive_bytes<const N: usize>(&mut self) -> ConcolicByteArray<N> {
        let v = self.constant(0u8);
        let mut array: [MaybeUninit<SymExpr>; N] = unsafe {
            MaybeUninit::uninit().assume_init()
        };

        for e in &mut array[..] {
            *e = MaybeUninit::new(v.clone());
        }

        // FIXME: when MaybeUninit::array_assume_init is stabilised
        let array = unsafe {
            (&array as *const _ as *const [SymExpr; N]).read()
        };

        ConcolicByteArray::from_symbolic(array)
    }

    pub fn zero(&mut self, bits: u32) -> SymExpr {
        self.consign.mk(Expr::Integer { value: 0, bits })
    }

    pub fn one(&mut self, bits: u32) -> SymExpr {
        self.consign.mk(Expr::Integer { value: 1, bits })
    }

    pub fn constant<T: Into<Expr>>(&mut self, t: T) -> SymExpr {
        self.consign.mk(t.into())
    }

    pub fn constant_with_size(&mut self, value: u64, bits: u32) -> SymExpr {
        self.consign.mk(Expr::Integer {value, bits})
    }

    pub fn collect(&mut self) {
        self.consign.collect();
        self.size_cache.clear();
    }

    pub fn variable<S: AsRef<str>>(&mut self, name: S, bits: u32) -> (Var, SymExpr) {
        let v = Var {
            name: name.as_ref().to_owned(),
            bits,
        };
        let e = self.consign.mk(Expr::Variable(v.clone()));
        (v, e)
    }

    pub fn unnamed_variable(&mut self, bits: u32) -> (u64, SymExpr) {
        let v = self.ivar_counter;
        let e = self.consign.mk(Expr::InternalVar(v, bits));
        self.ivar_counter += 1;
        (v, e)
    }

    // Size in bits
    pub fn size(&mut self, expr: &SymExpr) -> usize {
        use Expr::*;
        if let Some(sz) = self.size_cache.get(&expr.uid()) {
            *sz
        } else {
            match expr.get() {
                Bool(_) => 8,
                Integer { bits, .. } |
                Variable(Var { bits, .. }) |
                InternalVar(_, bits) => *bits as usize,

                Neg(e) => self.size(e),
                Not(e) => self.size(e),

                // Bitwise operation
                And(l, r) |
                Or(l, r) |
                Xor(l, r) |
                Add(l, r) |
                Sub(l, r) |
                Mul(l, r) |
                Div(l, r) |
                SDiv(l, r) |
                Rem(l, r) |
                SRem(l, r) |
                Shl(l, r) |
                Shr(l, r) |
                Sar(l, r) => self.size(l).max(self.size(r)),

                Carry(_, _) |
                SCarry(_, _) |
                SBorrow(_, _) |
                Less(_, _) |
                SLess(_, _) |
                LessEq(_, _) |
                SLessEq(_, _) |
                Greater(_, _) |
                SGreater(_, _) |
                GreaterEq(_, _) |
                SGreaterEq(_, _) |
                Equal(_, _) |
                NotEqual(_, _) => 8,

                Concat(l, r) => self.size(l) + self.size(r),
                Extract { expr, lsb, bits } => {
                    let esz = self.size(expr);
                    let off = lsb + bits;
                    if off > esz { esz - lsb } else { *bits }
                },
                SignExtend { bits, expr } |
                ZeroExtend { bits, expr } => self.size(expr) + *bits as usize,

                Relation { result, .. } => self.size(result),
            }
        }
    }

    pub fn concat(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.consign.mk(Expr::Concat(lhs, rhs))
    }

    pub fn extract(&mut self, expr: SymExpr, lsb: usize, bits: usize) -> SymExpr {
        self.consign.mk(Expr::Extract { expr, lsb, bits })
    }

    // Extend by X bits e.g extend 8 bits to 32 bits -> sign_extend(expr, 24)
    pub fn sign_extend(&mut self, expr: SymExpr, by_bits: u32) -> SymExpr {
        self.consign.mk(Expr::SignExtend { expr, bits: by_bits })
    }

    pub fn zero_extend(&mut self, expr: SymExpr, by_bits: u32) -> SymExpr {
        self.consign.mk(Expr::ZeroExtend { expr, bits: by_bits })
    }

    // Extend to X bits e.g extend 8 bits to 32 bits -> sign_extend(expr, 32)
    pub fn sign_extend_to(&mut self, expr: SymExpr, to_bits: u32) -> SymExpr {
        let size_orgi = self.size(&expr);
        let by_bits = to_bits - size_orgi as u32;
        self.consign.mk(Expr::SignExtend { expr, bits:by_bits })
    }

    pub fn zero_extend_to(&mut self, expr: SymExpr, to_bits: u32) -> SymExpr {
        let size_orgi = self.size(&expr);
        let by_bits = to_bits - size_orgi as u32;
        self.consign.mk(Expr::ZeroExtend { expr, bits:by_bits})
    }

    pub fn lift_signed2<F>(&mut self, lhs: SymExpr, rhs: SymExpr, f: F) -> SymExpr
    where F: FnOnce(&mut Self, SymExpr, SymExpr) -> SymExpr {
        let lsize = self.size(&lhs);
        let rsize = self.size(&rhs);

        let (lhs, rhs) = match lsize.cmp(&rsize) {
            Ordering::Less => (self.sign_extend_to(lhs, rsize as u32), rhs),
            Ordering::Greater => (lhs, self.sign_extend_to(rhs, lsize as u32)),
            Ordering::Equal => (lhs, rhs),
        };

        f(self, lhs, rhs)
    }

    pub fn lift_unsigned2<F>(&mut self, lhs: SymExpr, rhs: SymExpr, f: F) -> SymExpr
    where F: FnOnce(&mut Self, SymExpr, SymExpr) -> SymExpr {
        let lsize = self.size(&lhs);
        let rsize = self.size(&rhs);

        let (lhs, rhs) = match lsize.cmp(&rsize) {
            Ordering::Less => (self.zero_extend_to(lhs, rsize as u32), rhs),
            Ordering::Greater => (lhs, self.zero_extend_to(rhs, lsize as u32)),
            Ordering::Equal => (lhs, rhs),
        };

        f(self, lhs, rhs)
    }

    pub fn not(&mut self, expr: SymExpr) -> SymExpr {
        self.consign.mk(Expr::Not(expr))
    }

    pub fn and(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::And(lhs, rhs))
        })
    }

    pub fn or(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Or(lhs, rhs))
        })
    }

    pub fn xor(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Xor(lhs, rhs))
        })
    }

    pub fn neg(&mut self, expr: SymExpr) -> SymExpr {
        self.consign.mk(Expr::Neg(expr))
    }

    pub fn add(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Add(lhs, rhs))
        })
    }

    pub fn carry(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Carry(lhs, rhs))
        })
    }

    pub fn scarry(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::SCarry(lhs, rhs))
        })
    }

    pub fn sub(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Sub(lhs, rhs))
        })
    }

    pub fn sborrow(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::SBorrow(lhs, rhs))
        })
    }

    pub fn mul(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Mul(lhs, rhs))
        })
    }

    pub fn div(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Div(lhs, rhs))
        })
    }

    pub fn signed_div(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::SDiv(lhs, rhs))
        })
    }

    pub fn rem(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Rem(lhs, rhs))
        })
    }

    pub fn signed_rem(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::SRem(lhs, rhs))
        })
    }

    pub fn shift_left(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Shl(lhs, rhs))
        })
    }

    pub fn shift_right(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Shr(lhs, rhs))
        })
    }

    pub fn signed_shift_right(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Sar(lhs, rhs))
        })
    }

    pub fn equal(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Equal(lhs, rhs))
        })
    }

    pub fn not_equal(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::NotEqual(lhs, rhs))
        })
    }

    pub fn less(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Less(lhs, rhs))
        })
    }

    pub fn signed_less(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::SLess(lhs, rhs))
        })
    }

    pub fn less_eq(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::LessEq(lhs, rhs))
        })
    }

    pub fn signed_less_eq(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::SLessEq(lhs, rhs))
        })
    }

    pub fn greater(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::Greater(lhs, rhs))
        })
    }

    pub fn signed_greater(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::SGreater(lhs, rhs))
        })
    }

    pub fn greater_eq(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_unsigned2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::GreaterEq(lhs, rhs))
        })
    }

    pub fn signed_greater_eq(&mut self, lhs: SymExpr, rhs: SymExpr) -> SymExpr {
        self.lift_signed2(lhs, rhs, |slf, lhs, rhs| {
            slf.consign.mk(Expr::SGreaterEq(lhs, rhs))
        })
    }

    fn build_ast<'ctx>(ctx: &'ctx Z3Context, exprs: &mut HConMap<SymExpr, Z3BV<'ctx>>, expr: &SymExpr) -> Z3BV<'ctx> {
        if let Some(bv) = exprs.get(expr) {
            bv.clone()
        } else {
            use Expr::*;
            let bv = match expr.get() {
                // If it is a premitive type, then build from it
                Bool(v) => Z3BV::from_u64(ctx, if *v { 1 } else { 0 }, 8),
                Integer { value, bits } => Z3BV::from_u64(ctx, *value, *bits),
                Variable(Var { name, bits }) => Z3BV::new_const(ctx, name.as_ref(), *bits),
                InternalVar(_, bits) => Z3BV::fresh_const(ctx, "ivar", *bits),

                // Build recursively if it is an expression
                Not(e) => {
                    let exp = Self::build_ast(ctx, exprs, e);
                    exp.bvnot()
                },
                And(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvand(&rhs)
                },
                Or(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvor(&rhs)
                },
                Xor(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvxor(&rhs)
                },
                Neg(e) => {
                    let exp = Self::build_ast(ctx, exprs, e);
                    exp.bvneg()
                },
                Add(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvadd(&rhs)
                },
                Carry(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvadd_no_overflow(&rhs, false).ite(&t, &f)
                },
                SCarry(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvadd_no_overflow(&rhs, true).ite(&t, &f)
                },
                Sub(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvsub(&rhs)
                },
                SBorrow(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvsub_no_overflow(&rhs).ite(&t, &f)
                },
                Mul(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvmul(&rhs)
                },
                Div(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvudiv(&rhs)
                },
                SDiv(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvsdiv(&rhs)
                },
                Rem(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvurem(&rhs)
                },
                SRem(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvsrem(&rhs)
                },
                Shl(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvshl(&rhs)
                },
                Shr(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvlshr(&rhs)
                },
                Sar(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.bvashr(&rhs)
                },

                Less(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvult(&rhs).ite(&t, &f)
                },
                SLess(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvslt(&rhs).ite(&t, &f)
                },
                LessEq(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvule(&rhs).ite(&t, &f)
                },
                SLessEq(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvsle(&rhs).ite(&t, &f)
                },
                Greater(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvugt(&rhs).ite(&t, &f)
                },
                SGreater(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvsgt(&rhs).ite(&t, &f)
                },
                GreaterEq(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvuge(&rhs).ite(&t, &f)
                },
                SGreaterEq(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs.bvsge(&rhs).ite(&t, &f)
                },
                Equal(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs._eq(&rhs).ite(&t, &f)
                },
                NotEqual(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);

                    let t = Z3BV::from_u64(ctx, 1, 8);
                    let f = Z3BV::from_u64(ctx, 0, 8);

                    lhs._eq(&rhs).ite(&f, &t)
                },

                Concat(l, r) => {
                    let lhs = Self::build_ast(ctx, exprs, l);
                    let rhs = Self::build_ast(ctx, exprs, r);
                    lhs.concat(&rhs)
                },

                Extract { expr, lsb, bits } => {
                    let e = Self::build_ast(ctx, exprs, expr);
                    e.extract((*lsb + *bits - 1) as u32, *lsb as u32)
                },

                SignExtend { expr, bits } => {
                    let exp = Self::build_ast(ctx, exprs, expr);
                    exp.sign_ext(*bits as u32)
                },
                // extend by bits (original size + bits)
                ZeroExtend { expr, bits } => {
                    let exp = Self::build_ast(ctx, exprs, expr);
                    exp.zero_ext(*bits as u32)
                },

                Relation { result, .. } => {
                    // FIXME: for now, we model relations as a NO-OP (we do not build relations),
                    // and just emit the symbolic result
                    Self::build_ast(ctx, exprs, result)
                },
            };
            exprs.insert(expr.clone(), bv.clone());
            bv
        }
    }

    fn assert<'ctx>(ctx: &'ctx Z3Context, solver: &'ctx Z3Solver, exprs: &mut HConMap<SymExpr, Z3BV<'ctx>>, expr: &SymExpr) {
        use Expr::*;
        match expr.get() {
            Not(e) => {
                // NOTE: we should likely check that `e` is also a boolean expression
                let exp = Self::build_ast(&ctx, exprs, e);
                let zero = Z3BV::from_u64(&ctx, 0, exp.get_size());
                solver.assert(&exp._eq(&zero));
            },
            And(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                // In logical expression, only the last bit is used, so it's the same as bitwise operation
                let lone = Z3BV::from_u64(&ctx, 1, lhs.get_size());
                let rone = Z3BV::from_u64(&ctx, 1, rhs.get_size());

                // maybe not eq but >0 ?
                solver.assert(&Z3Bool::and(&ctx, &[&lhs._eq(&lone), &rhs._eq(&rone)]));
            },
            Or(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                let lone = Z3BV::from_u64(&ctx, 1, lhs.get_size());
                let rone = Z3BV::from_u64(&ctx, 1, rhs.get_size());

                solver.assert(&Z3Bool::or(&ctx, &[&lhs._eq(&lone), &rhs._eq(&rone)]));
            },
            Xor(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                let lone = Z3BV::from_u64(&ctx, 1, lhs.get_size());
                let rone = Z3BV::from_u64(&ctx, 1, rhs.get_size());

                solver.assert(&lhs._eq(&lone).xor(&rhs._eq(&rone)));
            },
            Less(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs.bvult(&rhs));
            },
            SLess(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs.bvslt(&rhs));
            },
            LessEq(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs.bvule(&rhs));
            },
            SLessEq(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs.bvsle(&rhs));
            },
            Greater(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs.bvugt(&rhs));
            },
            SGreater(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs.bvsgt(&rhs));
            },
            GreaterEq(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs.bvuge(&rhs));
            },
            SGreaterEq(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs.bvsge(&rhs));
            },
            Equal(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs._eq(&rhs));
            },
            NotEqual(l, r) => {
                let lhs = Self::build_ast(&ctx, exprs, l);
                let rhs = Self::build_ast(&ctx, exprs, r);

                solver.assert(&lhs._eq(&rhs).not());
            },
            x => unimplemented!("`{:?}` should not be used as a top-level query", x),
        }
    }

    pub fn solve_value<T: ByteCast, O: Order>(&mut self, expr: SymExpr, path_manager: &PathManager) -> Result<T, Error> {
        assert_eq!(T::SIZEOF, self.size(&expr));
        let (_, var) = self.unnamed_variable(T::SIZEOF as u32 * 8);
        let solve_var = var.clone();
        let solve_exp = self.equal(var, expr);
        if let Some(map) = self.solve(&solve_exp, vec![solve_var.clone()], path_manager) {
            if let Some(value) = map[&solve_var] {
                let buf = if O::NATIVE { value } else { value.swap_bytes() }.to_ne_bytes();
                Ok(T::from_bytes::<O>(&buf[..T::SIZEOF]))
            } else {
                let buf = [0u8; 8];
                Ok(T::from_bytes::<O>(&buf[..T::SIZEOF]))
            }
        } else {
            // Unsat
            Err(Error::UnSat)
        }
    }

    pub fn solve(&mut self, expr: &SymExpr, vars: Vec<SymExpr>, path_manager: &PathManager) -> Option<HashMap<SymExpr, Option<u64>>> {
        let ctx = Z3Context::new(&Z3Config::new());
        let solver = Z3Solver::new(&ctx);
        let mut exprs = HConMap::default();

        // taken contraints
        for taken in path_manager.path_constraints().map(|pc| pc.taken().constraint()) {
            Self::assert(&ctx, &solver, &mut exprs, taken);
        }

        Self::assert(&ctx, &solver, &mut exprs, expr);

        if solver.check() == z3::SatResult::Sat {
            if let Some(model) = solver.get_model() {
                Some(vars.into_iter().map(|var| {
                    let v = exprs.get(&var).and_then(|v| model.eval(v, true).and_then(|bv| bv.as_u64()));
                    (var, v)
                }).collect())
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct SymExprBuilder(Arc<RwLock<ExprBuilderImpl>>);

pub struct SymExprBuilderRef<'a>(parking_lot::RwLockReadGuard<'a, ExprBuilderImpl>);
impl<'a> Deref for SymExprBuilderRef<'a> {
    type Target = ExprBuilderImpl;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

pub struct SymExprBuilderMut<'a>(parking_lot::RwLockWriteGuard<'a, ExprBuilderImpl>);
impl<'a> Deref for SymExprBuilderMut<'a> {
    type Target = ExprBuilderImpl;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<'a> DerefMut for SymExprBuilderMut<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

impl SymExprBuilder {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(ExprBuilderImpl::new())))
    }

    pub fn as_ref<'a>(&'a self) -> SymExprBuilderRef<'a> {
        SymExprBuilderRef(self.0.read())
    }

    pub fn as_mut<'a>(&'a self) -> SymExprBuilderMut<'a> {
        SymExprBuilderMut(self.0.write())
    }
}
