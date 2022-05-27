use num_complex::Complex;
use std::{fmt, ops};

pub trait Integer: num_traits::PrimInt + ops::AddAssign + fmt::Display {}

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
    fn abs(self) -> f64;
}

impl Scalar for f64 {
    fn abs(self) -> f64 {
        self.abs()
    }
}

impl Scalar for f32 {
    fn abs(self) -> f64 {
        self.abs() as f64
    }
}

impl Scalar for Complex<f64> {
    fn abs(self) -> f64 {
        self.norm()
    }
}

impl Scalar for Complex<f32> {
    fn abs(self) -> f64 {
        self.norm() as f64
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

impl Nonzero for Complex<f64> {
    fn nonzero(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

impl Nonzero for Complex<f32> {
    fn nonzero(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

impl Nonzero for bool {
    fn nonzero(&self) -> bool {
        *self
    }
}
