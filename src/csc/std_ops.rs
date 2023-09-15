use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use std::ops::{Mul, Neg};

#[opimps::impl_uni_op(Neg)]
fn neg<I: Integer, T: Scalar>(self: CSC<I, T>) -> CSC<I, T> {
    CSC::new(
        self.rows(),
        self.cols(),
        self.rowidx,
        self.colptr,
        self.data.iter().map(|&a| -a).collect(),
    )
    .unwrap()
}

#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: CSC<I, T>) -> CSC<I, T> {
    self.mat_mat(&rhs).unwrap()
}

#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: CSR<I, T>) -> CSC<I, T> {
    self.mat_mat(&rhs.to_csc()).unwrap()
}
