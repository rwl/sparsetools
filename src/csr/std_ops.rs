use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use crate::{csr_add_csr, csr_sub_csr};
use num_complex::Complex64;
use std::ops::{Add, Mul, Neg, Sub};

#[opimps::impl_uni_op(Neg)]
fn neg<I: Integer, T: Scalar + Neg<Output = T>>(self: CSR<I, T>) -> CSR<I, T> {
    CSR::new(
        self.rows(),
        self.cols(),
        self.rowptr,
        self.colidx,
        self.values.iter().map(|&a| -a).collect(),
    )
    .unwrap()
}

#[opimps::impl_ops(Add)]
fn add<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: CSR<I, T>) -> CSR<I, T> {
    let nnz = self.nnz() + rhs.nnz();
    let mut rowptr = vec![I::zero(); self.rows() + 1];
    let mut colidx = vec![I::zero(); nnz];
    let mut values = vec![T::zero(); nnz];

    csr_add_csr(
        self.rows(),
        self.cols(),
        self.rowptr(),
        self.colidx(),
        self.values(),
        rhs.rowptr(),
        rhs.colidx(),
        rhs.values(),
        &mut rowptr,
        &mut colidx,
        &mut values,
    );

    CSR::new(self.rows(), self.cols(), rowptr, colidx, values).unwrap()
}

#[opimps::impl_ops(Sub)]
fn sub<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: CSR<I, T>) -> CSR<I, T> {
    let nnz = self.nnz() + rhs.nnz();
    let mut rowptr = vec![I::zero(); self.rows() + 1];
    let mut colidx = vec![I::zero(); nnz];
    let mut values = vec![T::zero(); nnz];

    csr_sub_csr(
        self.rows(),
        self.cols(),
        &self.rowptr(),
        &self.colidx(),
        &self.values(),
        &rhs.rowptr(),
        &rhs.colidx(),
        &rhs.values(),
        &mut rowptr,
        &mut colidx,
        &mut values,
    );

    CSR::new(self.rows(), self.cols(), rowptr, colidx, values).unwrap()
}

#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: CSR<I, T>) -> CSR<I, T> {
    self.mat_mat(&rhs).unwrap()
}

#[opimps::impl_ops_rprim(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: T) -> CSR<I, T> {
    CSR::new(
        self.rows(),
        self.cols(),
        self.rowptr().to_vec(),
        self.colidx().to_vec(),
        self.values().iter().map(|&v| v * rhs).collect::<Vec<T>>(),
    )
    .unwrap()
}
#[opimps::impl_ops_lprim(Mul)]
fn mul<I: Integer>(self: Complex64, rhs: CSR<I, Complex64>) -> CSR<I, Complex64> {
    CSR::new(
        rhs.rows(),
        rhs.cols(),
        rhs.rowptr().to_vec(),
        rhs.colidx().to_vec(),
        rhs.values()
            .iter()
            .map(|&v| self * v)
            .collect::<Vec<Complex64>>(),
    )
    .unwrap()
}
#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: &[T]) -> Vec<T> {
    self.mat_vec(rhs).unwrap()
}
#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: &Vec<T>) -> Vec<T> {
    self.mat_vec(rhs).unwrap()
}
