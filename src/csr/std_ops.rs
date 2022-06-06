use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use crate::{csr_add_csr, csr_sub_csr};
use std::ops::{Add, Mul, Sub};

#[opimps::impl_ops(Add)]
fn add<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: CSR<I, T>) -> CSR<I, T> {
    let nnz = self.nnz().to_usize().unwrap() + rhs.nnz().to_usize().unwrap();
    let mut rowptr = vec![I::zero(); self.rows.to_usize().unwrap() + 1];
    let mut colidx = vec![I::zero(); nnz];
    let mut data = vec![T::zero(); nnz];

    csr_add_csr(
        self.rows,
        self.cols,
        &self.rowptr,
        &self.colidx,
        &self.data,
        &rhs.rowptr,
        &rhs.colidx,
        &rhs.data,
        &mut rowptr,
        &mut colidx,
        &mut data,
    );

    CSR {
        rows: self.rows,
        cols: self.cols,
        rowptr,
        colidx,
        data,
    }
}

#[opimps::impl_ops(Sub)]
fn sub<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: CSR<I, T>) -> CSR<I, T> {
    let nnz = self.nnz().to_usize().unwrap() + rhs.nnz().to_usize().unwrap();
    let mut rowptr = vec![I::zero(); self.rows.to_usize().unwrap() + 1];
    let mut colidx = vec![I::zero(); nnz];
    let mut data = vec![T::zero(); nnz];

    csr_sub_csr(
        self.rows,
        self.cols,
        &self.rowptr,
        &self.colidx,
        &self.data,
        &rhs.rowptr,
        &rhs.colidx,
        &rhs.data,
        &mut rowptr,
        &mut colidx,
        &mut data,
    );

    CSR {
        rows: self.rows,
        cols: self.cols,
        rowptr,
        colidx,
        data,
    }
}

#[opimps::impl_ops(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: CSR<I, T>) -> CSR<I, T> {
    self.mat_mat(&rhs).unwrap()
}

#[opimps::impl_ops_rprim(Mul)]
fn mul<I: Integer, T: Scalar>(self: CSR<I, T>, rhs: T) -> CSR<I, T> {
    CSR {
        rows: self.rows,
        cols: self.cols,
        rowptr: self.rowptr.clone(),
        colidx: self.colidx.clone(),
        data: self.data.iter().map(|&v| v * rhs).collect::<Vec<T>>(),
    }
}

#[opimps::impl_ops_assign(std::ops::SubAssign)]
fn sub_assign<I: Integer, T: Scalar>(self: &mut CSR<I, T>, rhs: CSR<I, T>) {
    self.data
        .iter_mut()
        .zip(rhs.data.iter())
        .for_each(|(a, b)| *a -= *b)
}
