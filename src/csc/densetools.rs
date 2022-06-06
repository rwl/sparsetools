use crate::csc::CSC;
use crate::traits::{Integer, Scalar};
use densetools::arr::Arr;
use std::ops::Mul;

#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: Arr<T>) -> Arr<T> {
    Arr::with_vec(self.mat_vec(&rhs).unwrap())
}
