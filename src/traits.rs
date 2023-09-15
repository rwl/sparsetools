use pretty_dtoa::{dtoa, FmtFloatConfig};
use std::{fmt, ops};

use crate::util::DEFAULT_FLOAT_CONFIG;

pub trait Integer:
    num_traits::PrimInt + ops::AddAssign + ops::SubAssign + fmt::Display + fmt::Debug
{
}

impl Integer for usize {}
impl Integer for u8 {}
impl Integer for u16 {}
impl Integer for u32 {}
impl Integer for u64 {}
impl Integer for u128 {}

impl Integer for isize {}
impl Integer for i8 {}
impl Integer for i16 {}
impl Integer for i32 {}
impl Integer for i64 {}
impl Integer for i128 {}

pub trait Float: Scalar {}

impl Float for f64 {}
impl Float for f32 {}

pub trait Complex<F: Float>: Scalar {
    fn conj(&self) -> Self;
    fn real(&self) -> F;
    fn imag(&self) -> F;
}

impl Complex<f64> for num_complex::Complex<f64> {
    fn conj(&self) -> Self {
        self.conj()
    }

    fn real(&self) -> f64 {
        self.re
    }

    fn imag(&self) -> f64 {
        self.im
    }
}

impl Complex<f32> for num_complex::Complex<f32> {
    fn conj(&self) -> Self {
        self.conj()
    }

    fn real(&self) -> f32 {
        self.re
    }

    fn imag(&self) -> f32 {
        self.im
    }
}

pub trait Scalar<Output = Self>:
    PartialEq
    + Copy
    + Clone
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::Div<Output = Self>
    + ops::Neg<Output = Self>
    + ops::AddAssign
    + ops::SubAssign
    + ops::MulAssign
    + num_traits::Zero
    + num_traits::One<Output = Self>
    + fmt::Display
    + Nonzero
{
    fn pretty_string(&self, config: Option<FmtFloatConfig>) -> String;
}

impl Scalar for f64 {
    fn pretty_string(&self, config: Option<FmtFloatConfig>) -> String {
        dtoa(*self, config.unwrap_or(DEFAULT_FLOAT_CONFIG))
    }
}
impl Scalar for f32 {
    fn pretty_string(&self, config: Option<FmtFloatConfig>) -> String {
        dtoa(*self as f64, config.unwrap_or(DEFAULT_FLOAT_CONFIG))
    }
}

impl Scalar for num_complex::Complex<f64> {
    fn pretty_string(&self, config: Option<FmtFloatConfig>) -> String {
        format!(
            "{}{}j{}",
            dtoa(self.re, config.unwrap_or(DEFAULT_FLOAT_CONFIG)),
            if self.im.signum() < 0.0 { "-" } else { "+" },
            dtoa(self.im, config.unwrap_or(DEFAULT_FLOAT_CONFIG))
        )
        .to_string()
    }
}
impl Scalar for num_complex::Complex<f32> {
    fn pretty_string(&self, config: Option<FmtFloatConfig>) -> String {
        format!(
            "{}{}j{}",
            dtoa(self.re as f64, config.unwrap_or(DEFAULT_FLOAT_CONFIG)),
            if self.im.signum() < 0.0 { "-" } else { "+" },
            dtoa(self.im as f64, config.unwrap_or(DEFAULT_FLOAT_CONFIG))
        )
        .to_string()
    }
}

pub trait Nonzero {
    fn nonzero(&self) -> bool;
}

impl Nonzero for f64 {
    fn nonzero(&self) -> bool {
        *self != 0.0
    }
}

impl Nonzero for f32 {
    fn nonzero(&self) -> bool {
        *self != 0.0
    }
}

impl Nonzero for num_complex::Complex<f64> {
    fn nonzero(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

impl Nonzero for num_complex::Complex<f32> {
    fn nonzero(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

impl Nonzero for bool {
    fn nonzero(&self) -> bool {
        *self
    }
}
