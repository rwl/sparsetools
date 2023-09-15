use crate::coo::Coo;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use std::ops::Add;

#[opimps::impl_ops(Add)]
fn add<I: Integer, T: Scalar>(self: Coo<I, T>, rhs: Coo<I, T>) -> CSR<I, T> {
    assert_eq!(self.rows(), rhs.rows());
    assert_eq!(self.cols(), rhs.cols());

    let k = self.nnz();
    let nnz = k + rhs.nnz();

    let mut rowidx = Vec::with_capacity(nnz);
    let mut colidx = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    rowidx.extend(self.rowidx());
    colidx.extend(self.colidx());
    data.extend(self.data());

    rowidx.extend(rhs.rowidx());
    colidx.extend(rhs.colidx());
    data.extend(rhs.data());

    let a_mat = Coo::new(self.rows(), self.cols(), rowidx, colidx, data).unwrap();
    a_mat.to_csr() // Duplicate entries are summed.
}

#[opimps::impl_uni_op(std::ops::Neg)]
fn neg<I: Integer, T: Scalar>(self: Coo<I, T>) -> Coo<I, T> {
    Coo::new(
        self.rows(),
        self.cols(),
        self.rowidx,
        self.colidx,
        self.data.iter().map(|&d| -d).collect(),
    )
    .unwrap()
}