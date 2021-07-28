use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::ops::{RangeFull, RangeTo};

use fugue::bytes::{ByteCast, Order};

use either::Either;

use num_traits::cast::AsPrimitive;

use crate::builder::{Expr, SymExpr, SymExprBuilderMut};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ConcolicValue<T: ByteCast>(Either<SymExpr, T>);

impl<T: ByteCast + Debug> ConcolicValue<T> {
    pub fn unwrap_symbolic(self) -> SymExpr {
        self.0.unwrap_left()
    }

    pub fn unwrap_as_symbolic(&self) -> &SymExpr {
        self.0.as_ref().unwrap_left()
    }

    pub fn unwrap_concrete(self) -> T {
        self.0.unwrap_right()
    }

    pub fn unwrap_as_concrete(&self) -> &T {
        self.0.as_ref().unwrap_right()
    }
}

impl<T: ByteCast> ConcolicValue<T> {
    pub fn from_symbolic(t: SymExpr) -> Self {
        Self(Either::Left(t))
    }

    pub fn from_concrete(t: T) -> Self {
        Self(Either::Right(t))
    }

    pub fn is_concrete(&self) -> bool {
        self.0.is_right()
    }

    pub fn concrete(self) -> Option<T> {
        self.0.right()
    }

    pub fn map_concrete<U: ByteCast, F: FnMut(T) -> U>(self, f: F) -> ConcolicValue<U> {
        ConcolicValue(self.0.map_right(f))
    }

    pub fn is_symbolic(&self) -> bool {
        self.0.is_left()
    }

    pub fn symbolic(self) -> Option<SymExpr> {
        self.0.left()
    }

    pub fn map_symbolic<F: FnMut(SymExpr) -> SymExpr>(self, f: F) -> Self {
        Self(self.0.map_left(f))
    }

    pub fn map_concolic<U: ByteCast, F: FnMut(T) -> U, G: FnMut(SymExpr) -> SymExpr>(self, f: F, g: G) -> ConcolicValue<U> {
        self.map_concrete(f).map_symbolic(g)
    }

    pub fn unwrap_concolic<U, F: FnMut(T) -> U, G: FnMut(SymExpr) -> U>(self, f: F, g: G) -> U {
        let mut f = f;
        let mut g = g;
        match self.0 {
            Either::Right(concrete) => f(concrete),
            Either::Left(symbolic) => g(symbolic),
        }
    }

    pub fn lift2_concolic_full<U, F, G, H, I>(self, other: Self, builder: SymExprBuilderMut, f: F, g: G, h: H, i: I) -> U
    where F: FnMut(SymExprBuilderMut, T, T) -> U,
          G: FnMut(SymExprBuilderMut, T, SymExpr) -> U,
          H: FnMut(SymExprBuilderMut, SymExpr, T) -> U,
          I: FnMut(SymExprBuilderMut, SymExpr, SymExpr) -> U {

        let mut f = f;
        let mut g = g;
        let mut h = h;
        let mut i = i;

        match self.0 {
            Either::Right(c0) => match other.0 {
                Either::Right(c1) => f(builder, c0, c1),
                Either::Left(s1) => g(builder, c0, s1),
            },
            Either::Left(s0) => match other.0 {
                Either::Right(c1) => h(builder, s0, c1),
                Either::Left(s1) => i(builder, s0, s1),
            }
        }
    }

    pub fn try_unwrap_concolic<U, E, F: FnMut(T) -> Result<U, E>, G: FnMut(SymExpr) -> Result<U, E>>(self, f: F, g: G) -> Result<U, E> {
        let mut f = f;
        let mut g = g;
        match self.0 {
            Either::Right(concrete) => f(concrete),
            Either::Left(symbolic) => g(symbolic),
        }
    }

    pub fn cast<U: ByteCast + 'static, O: Order>(self, mut builder: SymExprBuilderMut) -> ConcolicValue<U>
    where T: AsPrimitive<U> {
        if T::SIZEOF > U::SIZEOF {
            // truncate
            self.map_concolic(
                |t| t.as_(),
                |t| if O::ENDIAN.is_big() {
                    builder.extract(t.clone(), (T::SIZEOF * 8) - (U::SIZEOF * 8), U::SIZEOF * 8)
                } else {
                    builder.extract(t.clone(), 0, U::SIZEOF * 8)
                })
        } else {
            // extend
            if U::SIGNED {
                // cast from unsigned to signed
                self.map_concolic(
                    |t| t.as_(),
                    |t| builder.sign_extend(t.clone(), (U::SIZEOF * 8) as u32))
            } else {
                self.map_concolic(
                    |t| t.as_(),
                    |t| builder.zero_extend(t.clone(), (U::SIZEOF * 8) as u32))
            }
        }
    }
}

impl<T> ConcolicValue<T>
where T: ByteCast, Expr: From<T> {
    pub fn lift2_concolic<U, F, G>(self, other: Self, mut builder: SymExprBuilderMut, f: F, g: G) -> U
    where F: FnMut(SymExprBuilderMut, T, T) -> U,
          G: FnMut(SymExprBuilderMut, SymExpr, SymExpr) -> U {

        let mut f = f;
        let mut g = g;

        match self.0 {
            Either::Right(c0) => match other.0 {
                Either::Right(c1) => f(builder, c0, c1),
                Either::Left(s1) => { let s0 = builder.constant(c0); g(builder, s0, s1) },
            },
            Either::Left(s0) => match other.0 {
                Either::Right(c1) => { let s1 = builder.constant(c1); g(builder, s0, s1) },
                Either::Left(s1) => g(builder, s0, s1),
            }
        }
    }

    pub fn into_symbolic(self, mut builder: SymExprBuilderMut) -> SymExpr {
        match self.0 {
            Either::Right(c) => builder.constant(c),
            Either::Left(s) => s,
        }
    }
}

impl<T: ByteCast> From<T> for ConcolicValue<T> {
    fn from(t: T) -> Self {
        Self(Either::Right(t))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ConcolicByteArray<const N: usize>(Either<[SymExpr; N], [u8; N]>);

impl<const N: usize> ConcolicByteArray<N> {
    pub fn concrete_primitive() -> ConcolicByteArray<N> {
        ConcolicByteArray::from_concrete([0u8; N])
    }

    pub fn symbolic_primitive(builder: &mut SymExprBuilderMut) -> ConcolicByteArray<N> {
        builder.primitive_bytes::<{ N }>()
    }

    pub fn from_concrete(array: [u8; N]) -> Self {
        Self(Either::Right(array))
    }

    pub fn from_symbolic(array: [SymExpr; N]) -> Self {
        Self(Either::Left(array))
    }

    pub fn is_concrete(&self) -> bool {
        self.0.is_right()
    }

    pub fn concrete(self) -> Option<[u8; N]> {
        self.0.right()
    }

    pub fn unwrap_concrete(self) -> [u8; N] {
        self.0.unwrap_right()
    }

    pub fn is_symbolic(&self) -> bool {
        self.0.is_left()
    }

    pub fn symbolic(self) -> Option<[SymExpr; N]> {
        self.0.left()
    }

    pub fn unwrap_symbolic(self) -> [SymExpr; N] {
        self.0.unwrap_left()
    }

    pub fn as_slice_to<'a>(&'a self, range: RangeTo<usize>) -> ConcolicByteSlice<'a> {
        ConcolicByteSlice(self.0.as_ref()
                          .map_left(|v| v.index(range))
                          .map_right(|v| v.index(range)))
    }

    pub fn as_slice_to_mut<'a>(&'a mut self, range: RangeTo<usize>) -> ConcolicByteSliceMut<'a> {
        ConcolicByteSliceMut(self.0.as_mut()
                          .map_left(|v| v.index_mut(range))
                          .map_right(|v| v.index_mut(range)))
    }

    pub fn as_slice<'a>(&'a self) -> ConcolicByteSlice<'a> {
        ConcolicByteSlice(self.0.as_ref()
                          .map_left(|v| v.index(RangeFull))
                          .map_right(|v| v.index(RangeFull)))
    }

    pub fn as_slice_mut<'a>(&'a mut self) -> ConcolicByteSliceMut<'a> {
        ConcolicByteSliceMut(self.0.as_mut()
                          .map_left(|v| v.index_mut(RangeFull))
                          .map_right(|v| v.index_mut(RangeFull)))
    }

    pub const fn len(&self) -> usize {
        N
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ConcolicByteSlice<'a>(Either<&'a [SymExpr], &'a [u8]>);

impl<'a> ConcolicByteSlice<'a> {
    pub fn is_concrete(&self) -> bool {
        self.0.is_right()
    }

    pub fn concrete(self) -> Option<&'a [u8]> {
        self.0.right()
    }

    pub fn unwrap_concrete(self) -> &'a [u8] {
        self.0.unwrap_right()
    }

    pub fn is_symbolic(&self) -> bool {
        self.0.is_left()
    }

    pub fn symbolic(self) -> Option<&'a [SymExpr]> {
        self.0.left()
    }

    pub fn unwrap_symbolic(self) -> &'a [SymExpr] {
        self.0.unwrap_left()
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ConcolicByteSliceMut<'a>(Either<&'a mut [SymExpr], &'a mut [u8]>);

impl<'a> ConcolicByteSliceMut<'a> {
    pub fn is_concrete(&self) -> bool {
        self.0.is_right()
    }

    pub fn concrete(self) -> Option<&'a mut [u8]> {
        self.0.right()
    }

    pub fn unwrap_concrete(self) -> &'a mut [u8] {
        self.0.unwrap_right()
    }

    pub fn is_symbolic(&self) -> bool {
        self.0.is_left()
    }

    pub fn symbolic(self) -> Option<&'a mut [SymExpr]> {
        self.0.left()
    }

    pub fn unwrap_symbolic(self) -> &'a mut [SymExpr] {
        self.0.unwrap_left()
    }
}

impl<'a> Into<ConcolicByteSlice<'a>> for ConcolicByteSliceMut<'a> {
    fn into(self) -> ConcolicByteSlice<'a> {
        ConcolicByteSlice(self.0
                          .map_left(|v| v as &[_])
                          .map_right(|v| v as &[_]))
    }
}
