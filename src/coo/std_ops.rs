use crate::coo::Coo;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use std::ops::{Add, Neg};

#[opimps::impl_ops(Add)]
fn add<I: Integer, T: Scalar>(self: Coo<I, T>, rhs: Coo<I, T>) -> CSR<I, T> {
    assert_eq!(self.rows(), rhs.rows());
    assert_eq!(self.cols(), rhs.cols());

    let k = self.nnz();
    let nnz = k + rhs.nnz();

    let mut rowidx = Vec::with_capacity(nnz);
    let mut colidx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    rowidx.extend(self.rowidx());
    colidx.extend(self.colidx());
    values.extend(self.values());

    rowidx.extend(rhs.rowidx());
    colidx.extend(rhs.colidx());
    values.extend(rhs.values());

    let a_mat = Coo::new(self.rows(), self.cols(), rowidx, colidx, values).unwrap();
    a_mat.to_csr() // Duplicate entries are summed.
}

#[opimps::impl_uni_op(Neg)]
fn neg<I: Integer, T: Scalar + Neg<Output = T>>(self: Coo<I, T>) -> Coo<I, T> {
    Coo::new(
        self.rows(),
        self.cols(),
        self.rowidx,
        self.colidx,
        self.values.iter().map(|&d| -d).collect(),
    )
    .unwrap()
}
