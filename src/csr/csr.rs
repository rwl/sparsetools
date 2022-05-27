use crate::coo::Coo;
use crate::csc::CSC;
use crate::row::{
    csr_diagonal, csr_has_canonical_format, csr_has_sorted_indices, csr_matmat, csr_matmat_maxnnz,
    csr_matvec, csr_select, csr_sort_indices, csr_sum_duplicates, csr_tocsc, csr_todense,
    expandptr,
};
use crate::traits::{Integer, Scalar};
use std::cmp::min;

/// A sparse matrix with scalar values stored in Compressed Sparse Row (CSR) format.
pub struct CSR<I: Integer, T: Scalar> {
    pub rows: I,
    pub cols: I,
    /// Row pointers (size rows+1).
    pub rowptr: Vec<I>,
    /// Column indexes (size nnz).
    pub colidx: Vec<I>,
    /// Explicitly stored values (size nnz).
    pub data: Vec<T>,
}

impl<I: Integer, T: Scalar> CSR<I, T> {
    /// Creates a new CSR matrix. Inputs are not copied. An error
    /// is returned if the slice arguments do not have the correct length.
    pub fn new(
        rows: I,
        cols: I,
        rowptr: Vec<I>,
        colidx: Vec<I>,
        data: Vec<T>,
    ) -> Result<Self, String> {
        if rowptr.len() != rows.to_usize().unwrap() + 1 {
            return Err("rowptr has invalid length".to_string());
        }
        let nnz = rowptr[rowptr.len() - 1].to_usize().unwrap();
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
            rowptr[i] = I::from(i).unwrap();
        }
        let mut colidx = vec![I::zero(); n];
        for i in 0..n {
            colidx[i] = I::from(i).unwrap();
        }
        CSR {
            rows: I::from(n).unwrap(),
            cols: I::from(n).unwrap(),
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
        csr_has_canonical_format::<I>(
            I::from(self.rowptr.len() - 1).unwrap(),
            &self.rowptr,
            &self.colidx,
        )
    }

    /// HasSortedIndexes returns true if the indexes of the matrix are in
    /// sorted order.
    pub fn has_sorted_indexes(&self) -> bool {
        csr_has_sorted_indices::<I>(
            I::from(self.rowptr.len() - 1).unwrap(),
            &self.rowptr,
            &self.colidx,
        )
    }

    /// SortIndexes sorts the indexes of this matrix in place.
    pub fn sort_indexes(&mut self) {
        csr_sort_indices::<I, T>(
            I::from(self.rowptr.len() - 1).unwrap(),
            &self.rowptr,
            &mut self.colidx,
            &mut self.data,
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
        if self.rowptr.len() != self.rows.to_usize().unwrap() + 1 {
            return Err("index pointer has invalid length".to_string());
        }
        let nnz = self.nnz().to_usize().unwrap();
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
    pub fn mat_vec(&self, x: &[T]) -> Result<Vec<T>, String> {
        if x.len() != self.cols.to_usize().unwrap() {
            return Err(format!("len(x) ({}) != mat.cols ({})", x.len(), self.cols));
        }
        let mut result = vec![T::zero(); self.rows.to_usize().unwrap()];
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

        let mut rowptr = vec![I::zero(); rows.to_usize().unwrap() + 1];

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
            // self.sum_duplicates();
            return Err("must have canonical form, sum_duplicates".to_string());
        }

        let mut rowidx_full = Vec::new();
        let rowidx = match rowidx {
            None => {
                rowidx_full.resize(self.rows.to_usize().unwrap(), I::zero());
                for i in 0..rowidx_full.len() {
                    rowidx_full[i] = I::from(i).unwrap();
                }
                rowidx_full.as_slice()
            }
            Some(rowidx) => {
                for &ri in rowidx {
                    if ri < I::zero() || ri >= self.rows {
                        return Err(format!("out of range: {}", ri));
                    }
                }
                rowidx
            }
        };

        let mut colidx_full = Vec::new();
        let colidx = match colidx {
            None => {
                colidx_full.resize(self.cols.to_usize().unwrap(), I::zero());
                for i in 0..colidx_full.len() {
                    colidx_full[i] = I::from(i).unwrap();
                }
                colidx_full.as_slice()
            }
            Some(colidx) => {
                for &ci in colidx {
                    if ci < I::zero() || ci >= self.cols {
                        return Err(format!("out of range: {}", ci));
                    }
                }
                colidx
            }
        };

        let mut Bp: Vec<I> = vec![];
        let mut Bj: Vec<I> = vec![];
        let mut Bx: Vec<T> = vec![];

        csr_select(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.data,
            &rowidx,
            &colidx,
            &mut Bp,
            &mut Bj,
            &mut Bx,
        );

        CSR::new(
            I::from(rowidx.len()).unwrap(),
            I::from(colidx.len()).unwrap(),
            Bp,
            Bj,
            Bx,
        )
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
        let mut colptr = vec![I::zero(); self.cols.to_usize().unwrap() + 1];
        let mut rowidx = vec![I::zero(); self.nnz().to_usize().unwrap()];
        let mut data = vec![T::zero(); self.nnz().to_usize().unwrap()];

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
        let rows = self.rows.to_usize().unwrap();
        let cols = self.cols.to_usize().unwrap();
        // let mut dense = vec![&mut vec![T::zero(); cols]; rows];
        let mut dense = vec![vec![T::zero(); cols]; rows];
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
        let size = min(self.rows, self.cols).to_usize().unwrap();
        let mut diag = vec![T::zero(); size];
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
