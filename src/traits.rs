use std::{fmt, ops};

pub trait Integer: num_traits::PrimInt + ops::AddAssign + fmt::Display {}

impl Integer for usize {}
impl Integer for isize {}

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
    + ops::AddAssign
    + ops::SubAssign
    + ops::MulAssign
    + num_traits::Zero
    + num_traits::One<Output = Self>
    + fmt::Display
    + Nonzero
{
}

impl Scalar for f64 {}
impl Scalar for f32 {}

impl Scalar for num_complex::Complex<f64> {}
impl Scalar for num_complex::Complex<f32> {}

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
