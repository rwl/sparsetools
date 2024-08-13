use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use crate::{csr_add_csr, csr_sub_csr};
use num_complex::Complex64;
use std::ops::{Add, Mul, Neg, Sub};

#[opimps::impl_uni_op(Neg)]
fn neg<I: Integer, T: Scalar + Neg<Output = T>>(self: CSC<I, T>) -> CSC<I, T> {
    CSC::new(
        self.rows(),
        self.cols(),
        self.rowidx,
        self.colptr,
        self.values.iter().map(|&a| -a).collect(),
    )
    .unwrap()
}

#[opimps::impl_ops(Add)]
fn add<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: CSC<I, T>) -> CSC<I, T> {
    let nnz = self.nnz() + rhs.nnz();
    let mut colptr = vec![I::zero(); self.cols() + 1];
    let mut rowidx = vec![I::zero(); nnz];
    let mut values = vec![T::zero(); nnz];

    csr_add_csr(
        self.cols(),
        self.rows(),
        self.colptr(),
        self.rowidx(),
        self.values(),
        rhs.colptr(),
        rhs.rowidx(),
        rhs.values(),
        &mut colptr,
        &mut rowidx,
        &mut values,
    );

    CSC::new(self.rows(), self.cols(), rowidx, colptr, values).unwrap()
}

#[opimps::impl_ops(Sub)]
fn sub<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: CSC<I, T>) -> CSC<I, T> {
    let nnz = self.nnz() + rhs.nnz();
    let mut colptr = vec![I::zero(); self.cols() + 1];
    let mut rowidx = vec![I::zero(); nnz];
    let mut values = vec![T::zero(); nnz];

    csr_sub_csr(
        self.cols(),
        self.rows(),
        &self.colptr(),
        &self.rowidx(),
        &self.values(),
        &rhs.colptr(),
        &rhs.rowidx(),
        &rhs.values(),
        &mut colptr,
        &mut rowidx,
        &mut values,
    );

    CSC::new(self.rows(), self.cols(), colptr, rowidx, values).unwrap()
}

#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: CSC<I, T>) -> CSC<I, T> {
    self.mat_mat(&rhs).unwrap()
}

#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: CSR<I, T>) -> CSC<I, T> {
    self.mat_mat(&rhs.to_csc()).unwrap()
}

#[opimps::impl_ops_rprim(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: T) -> CSC<I, T> {
    CSC::new(
        self.rows(),
        self.cols(),
        self.colptr().to_vec(),
        self.rowidx().to_vec(),
        self.values().iter().map(|&v| v * rhs).collect::<Vec<T>>(),
    )
    .unwrap()
}
#[opimps::impl_ops_lprim(Mul)]
fn mul<I: Integer>(self: Complex64, rhs: CSC<I, Complex64>) -> CSC<I, Complex64> {
    CSC::new(
        rhs.rows(),
        rhs.cols(),
        rhs.colptr().to_vec(),
        rhs.rowidx().to_vec(),
        rhs.values()
            .iter()
            .map(|&v| self * v)
            .collect::<Vec<Complex64>>(),
    )
    .unwrap()
}
#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: &[T]) -> Vec<T> {
    self.mat_vec(rhs).unwrap()
}
#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSC<I, T>, rhs: &Vec<T>) -> Vec<T> {
    self.mat_vec(rhs).unwrap()
}
