use crate::coo::Coo;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use std::ops::Add;

impl<I, T> Add<Coo<I, T>> for Coo<I, T>
where
    I: Integer,
    T: Scalar,
{
    type Output = CSR<I, T>;

    /// Creates a CSR matrix that is the element-wise sum of the
    /// receiver and the given matrix.
    fn add(self, rhs: Coo<I, T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let k = self.nnz();
        let nnz = (k + rhs.nnz()).to_usize().unwrap();

        let mut rowidx = Vec::with_capacity(nnz);
        let mut colidx = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);

        rowidx.extend(&self.rowidx);
        colidx.extend(&self.colidx);
        data.extend(&self.data);

        rowidx.extend(&rhs.rowidx);
        colidx.extend(&rhs.colidx);
        data.extend(&rhs.data);

        let a_mat = Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colidx,
            data,
        };
        a_mat.to_csr() // Duplicate entries are summed.
    }
}
