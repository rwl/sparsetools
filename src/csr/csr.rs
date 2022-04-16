use crate::coo::Coo;
use crate::csc::CSC;
use crate::scalar::Scalar;
use crate::{
    csr_diagonal, csr_has_canonical_format, csr_has_sorted_indices, csr_matmat, csr_matmat_maxnnz,
    csr_matvec, csr_select, csr_sort_indices, csr_sum_duplicates, csr_tocsc, csr_todense,
    expandptr, get_csr_submatrix,
};
use num_traits::PrimInt;
use std::cmp::min;

/// A sparse matrix with scalar values stored in Compressed Sparse Row (CSR) format.
pub struct CSR<I: PrimInt, T: Scalar> {
    pub rows: I,
    pub cols: I,
    /// Row pointers (size rows+1).
    pub rowptr: Vec<I>,
    /// Column indexes (size nnz).
    pub colidx: Vec<I>,
    /// Explicitly stored values (size nnz).
    pub data: Vec<T>,
}

impl<I: PrimInt, T: Scalar> CSR<I, T> {
    /// Creates a new CSR matrix. Inputs are not copied. An error
    /// is returned if the slice arguments do not have the correct length.
    pub fn new(
        rows: I,
        cols: I,
        rowptr: Vec<I>,
        colidx: Vec<I>,
        data: Vec<T>,
    ) -> Result<Self, String> {
        if rowptr.len() != rows + 1 {
            return Err("rowptr has invalid length".to_string());
        }
        let nnz = rowptr[rowptr.len() - 1];
        if colidx.len() < nnz {
            return Err("colidx has fewer than nnz elements".to_string());
        }
        if data.len() < nnz {
            return Err("data array has fewer than nnz elements".to_string());
        }
        Ok(CSR {
            rows,
            cols,
            rowptr,
            colidx,
            data,
        })
    }

    /// Creates a new CSR matrix with the given values on the main
    /// diagonal. The input is not copied.
    pub fn from_diag(data: Vec<T>) -> Self {
        let n = data.len();
        let mut rowptr = vec![I::zero(); n + 1];
        for i in 0..n + 1 {
            rowptr[i] = i;
        }
        let mut colidx = vec![I::zero(); n];
        for i in 0..n {
            colidx[i] = i;
        }
        CSR {
            rows: n,
            cols: n,
            rowptr,
            colidx,
            data,
        }
    }

    /// Returns the count of explicitly-stored values (nonzeros).
    pub fn nnz(&self) -> I {
        if self.rowptr.is_empty() {
            I::zero()
        } else {
            self.rowptr[self.rowptr.len() - 1]
        }
    }

    /// Creates a CSC matrix that is the transpose of the receiver.
    /// The underlying index and data slices are not copied.
    pub fn transpose(self) -> CSC<I, T> {
        CSC {
            rows: self.cols,
            cols: self.rows,
            rowidx: self.colidx,
            colptr: self.rowptr,
            data: self.data,
        }
    }

    /// An alias for `transpose`.
    pub fn t(self) -> CSC<I, T> {
        self.transpose()
    }

    /// Returns true if the matrix has sorted indexes and no duplicates.
    pub fn has_canonical_format(&self) -> bool {
        csr_has_canonical_format(self.rowptr.len() - 1, &self.rowptr, &self.colidx)
    }

    /// HasSortedIndexes returns true if the indexes of the matrix are in
    /// sorted order.
    pub fn has_sorted_indexes(&self) -> bool {
        csr_has_sorted_indices(self.rowptr.len() - 1, &self.rowptr, &self.colidx)
    }

    /// SortIndexes sorts the indexes of this matrix in place.
    pub fn sort_indexes(&self) {
        csr_sort_indices(
            self.rowptr.len() - 1,
            &self.rowptr,
            &self.colidx,
            &self.data,
        )
    }

    /// Sums duplicate entries.
    pub fn sum_duplicates(&mut self) -> Result<(), String> {
        if self.has_canonical_format() {
            return Ok(());
        }
        self.sort_indexes();

        csr_sum_duplicates(
            self.rows,
            self.cols,
            &mut self.rowptr,
            &mut self.colidx,
            &mut self.data,
        );

        self.prune() // nnz may have changed
    }

    /// Prune removes empty space after all non-zero elements.
    pub fn prune(&mut self) -> Result<(), String> {
        if self.rowptr.len() != self.rows + 1 {
            return Err("index pointer has invalid length".to_string());
        }
        let nnz = self.nnz();
        if self.colidx.len() < nnz {
            return Err("indexes array has fewer than nnz elements".to_string());
        }
        if self.data.len() < nnz {
            return Err("data array has fewer than nnz elements".to_string());
        }

        // self.data = self.data.[..nnz];
        self.data.resize(nnz, T::zero());
        // self.colidx = self.colidx[..nnz];
        self.colidx.resize(nnz, I::zero());

        Ok(())
    }

    /// Performs matrix-vector multiplication. The length of x must be equal
    /// to the number of columns of the receiver.
    pub fn mat_vec(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != self.cols {
            return Err(format!("len(x) ({}) != mat.cols ({})", x.len(), self.cols));
        }
        let mut result = vec![0.0; self.rows];
        csr_matvec(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.data,
            x,
            &mut result,
        );
        Ok(result)
    }

    /// Performs matrix-matrix multiplication. The number of rows of x
    /// must equal the number of columns of the receiver.
    pub fn mat_mat(&self, x: &CSR<I, T>) -> Result<CSR<I, T>, String> {
        if self.cols != x.rows {
            return Err(format!(
                "dimension mismatch, rows {} cols {}",
                x.rows, self.cols
            ));
        }
        let rows = self.rows;
        let cols = x.cols;

        let mut rowptr = vec![0; rows + 1];

        let nnz = csr_matmat_maxnnz(rows, cols, &self.rowptr, &self.colidx, &x.rowptr, &x.colidx); //, &rowptr);

        let mut colidx = vec![I::zero(); nnz];
        let mut data = vec![T::zero(); nnz];

        csr_matmat(
            rows,
            cols,
            &self.rowptr,
            &self.colidx,
            &self.data,
            &x.rowptr,
            &x.colidx,
            &x.data,
            &mut rowptr,
            &mut colidx,
            &mut data,
        );

        CSR::new(rows, cols, rowptr, colidx, data)
    }

    /// Creates a new CSR matrix with only the selected rows and columns
    /// of the receiver.
    pub fn select(&self, rowidx: Option<&[I]>, colidx: Option<&[I]>) -> Result<CSR<I, T>, String> {
        if !self.has_canonical_format() {
            self.sum_duplicates();
        }

        let rowidx = match rowidx {
            None => {
                let mut rowidx = vec![I::zero(); self.rows];
                for i in 0..rowidx.len() {
                    rowidx[i] = i;
                }
                rowidx
            }
            Some(rowidx) => rowidx,
        };
        let colidx = match colidx {
            None => {
                let mut colidx = vec![I::zero(); self.cols];
                for i in 0..colidx.len() {
                    colidx[i] = i;
                }
            }
            Some(colidx) => colidx,
        };

        for ri in rowidx {
            if ri < 0 || ri >= self.rows {
                return Err(format!("out of range: {}", ri));
            }
        }
        for ci in colidx {
            if ci < 0 || ci >= self.cols {
                return Err(format!("out of range: {}", ci));
            }
        }

        let mut Bp: Vec<I> = vec![];
        let mut Bj: Vec<I> = vec![];
        let mut Bx: Vec<T> = vec![];

        csr_select(
            self.rows,
            self.cols,
            self.rowptr,
            self.colidx,
            self.data,
            &rowidx,
            &colidx,
            &mut Bp,
            &mut Bj,
            &mut Bx,
        );

        CSR::new(rowidx.len(), colidx.len(), Bp, Bj, Bx)
    }

    /// Converts the matrix into Coordinate (Coo) format.
    pub fn to_coo(&self) -> Coo<I, T> {
        // let data = vec![T::zero(); self.data.len()];
        // copy(data, self.data);
        let data = self.data.clone();

        // let colidx = vec![I::zero(); self.colidx.len()];
        // copy(colidx, self.colidx);
        let colidx = self.colidx.clone();

        let mut rowidx = vec![I::zero(); self.colidx.len()];

        expandptr(self.rows, &self.rowptr, &mut rowidx);

        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colidx,
            data,
        }
    }

    /// Converts the matrix to Compressed Sparse Column (CSC) format.
    pub fn to_csc(&self) -> CSC<I, T> {
        let mut colptr = vec![I::zero(); self.cols + 1];
        let mut rowidx = vec![T::zero(); self.nnz()];
        let mut data = vec![T::zero(); self.nnz()];

        csr_tocsc(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.data,
            &mut colptr,
            &mut rowidx,
            &mut data,
        );

        CSC {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colptr,
            data,
        }
    }

    /// Converts the matrix into a dense 2D slice.
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let mut dense = vec![vec![T::zero(); self.cols], self.rows];
        // for r := range dense {
        // 	dense[r] = make([]float64, self.cols)
        // }
        csr_todense(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.data,
            &mut dense,
        );
        dense
    }

    /// Returns a vector of the elements on the main diagonal.
    pub fn diagonal(&self) -> Vec<T> {
        let mut diag = vec![T::zero(); min(self.rows, self.cols)];
        csr_diagonal(
            0,
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.data,
            &mut diag,
        );
        diag
    }
}
