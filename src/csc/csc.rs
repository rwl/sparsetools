use crate::coo::Coo;
use crate::csr::CSR;
use crate::scalar::Scalar;
use crate::{
    csc_matmat, csc_matmat_maxnnz, csc_matvec, csc_tocsr, csr_diagonal, csr_has_canonical_format,
    csr_has_sorted_indices, csr_matmat, csr_matmat_maxnnz, csr_select, csr_sort_indices,
    csr_sum_duplicates, csr_tocoo, csr_tocsc,
};
use num_traits::PrimInt;
use std::cmp::min;

/// A sparse matrix with scalar values stored in Compressed Sparse Column (CSC) format.
pub struct CSC<I: PrimInt, T: Scalar> {
    pub rows: I,
    pub cols: I,
    /// Row indexes (size nnz).
    pub rowidx: Vec<I>,
    /// Column pointers (size cols+1).
    pub colptr: Vec<I>,
    /// Explicitly stored values (size nnz).
    pub data: Vec<T>,
}

impl<I: PrimInt, T: Scalar> CSC<I, T> {
    /// Creates a new CSC matrix. Inputs are not copied. An error
    /// is returned if the slice arguments do not have the correct length.
    pub fn new(
        rows: I,
        cols: I,
        rowidx: Vec<I>,
        colptr: Vec<I>,
        data: Vec<T>,
    ) -> Result<Self, String> {
        if colptr.len() != cols + 1 {
            return Err("colptr has invalid length".to_string());
        }
        let nnz = colptr[colptr.len() - 1];
        if rowidx.len() < nnz {
            return Err("rowidx array has fewer than nnz elements".to_string());
        }
        if data.len() < nnz {
            return Err("data array has fewer than nnz elements".to_string());
        }
        Ok(Self {
            rows,
            cols,
            rowidx,
            colptr,
            data,
        })
    }

    /// Creates a new CSC matrix with the given values on the main
    /// diagonal. The input is not copied.
    pub fn from_diag(data: Vec<T>) -> Self {
        let n = data.len();
        let mut rowidx = vec![I::zero(); n];
        for i in 0..rowidx.len() {
            rowidx[i] = i;
        }
        let mut colptr = vec![I::zero(); n + 1];
        for i in 0..colptr.len() {
            colptr[i] = i;
        }
        Self {
            rows: n,
            cols: n,
            rowidx,
            colptr,
            data,
        }
    }

    pub fn shape(&self) -> (I, I) {
        (self.rows, self.cols)
    }

    /// Returns the count of explicitly stored values (nonzeros).
    pub fn nnz(&self) -> I {
        if self.colptr.len() == 0 {
            0
        } else {
            self.colptr[self.colptr.len() - 1]
        }
    }

    /// Copy creates an identical CSC matrix with newly allocated
    /// index, pointer and data slices.
    pub fn copy(&self) -> CSC<I, T> {
        // let rowidx = vec!(I::zero(); self.rowidx.len());
        // let colptr = vec!(I::zero(); self.colptr.len());
        // let data = vec!(T::zero(); self.data.len());
        // copy(rowidx, self.rowidx);
        // copy(colptr, self.colptr);
        // copy(data, self.data);

        CSC {
            rows: self.rows,
            cols: self.cols,
            rowidx: self.rowidx.clone(),
            colptr: self.colptr.clone(),
            data: self.data.clone(),
        }
    }

    /// Transpose creates a CSR matrix that is the transpose of the
    /// receiver.
    pub fn transpose(&self) -> CSR<I, T> {
        CSR {
            rows: self.cols,
            cols: self.rows,
            rowptr: self.colptr.clone(),
            colidx: self.rowidx.clone(),
            data: self.data.clone(),
        }
    }

    /// An alias for Transpose.
    pub fn t(&self) -> CSR<I, T> {
        self.transpose()
    }

    /// Returns true if the matrix has sorted indexes and no duplicates.
    pub fn has_canonical_format(&self) -> bool {
        csr_has_canonical_format(self.colptr.len() - 1, &self.colptr, &self.rowidx)
    }

    /// Returns true if the indexes of the matrix are in sorted order.
    pub fn has_sorted_indexes(&self) -> bool {
        csr_has_sorted_indices(self.colptr.len() - 1, &self.colptr, &self.rowidx)
    }

    /// Sorts the indexes of this matrix in place.
    pub fn sort_indexes(&self) {
        csr_sort_indices(
            self.colptr.len() - 1,
            &self.colptr,
            &self.rowidx,
            &self.data,
        )
    }

    /// Sums duplicate entries.
    pub fn sum_duplicates(&mut self) {
        if self.has_canonical_format() {
            return;
        }
        self.sort_indexes();

        csr_sum_duplicates(self.cols, self.rows, &self.colptr, &self.rowidx, &self.data);

        self.prune(); // nnz may have changed
    }

    /// Removes empty space after all non-zero elements.
    pub fn prune(&mut self) -> Result<(), String> {
        if self.colptr.len() != self.cols + 1 {
            return Err("index pointer has invalid length".to_string());
        }
        let nnz = self.nnz();
        if self.rowidx.len() < nnz {
            return Err("indexes array has fewer than nnz elements".to_string());
        }
        if self.data.len() < nnz {
            return Err("data array has fewer than nnz elements".to_string());
        }

        self.data = self.data[..nnz];
        self.rowidx = self.rowidx[..nnz];
        Ok(())
    }

    /// Performs matrix-vector multiplication. The length of x must be equal
    /// to the number of columns of the receiver.
    pub fn mat_vec(&self, x: &[T]) -> Result<Vec<T>, String> {
        if x.len() != self.cols {
            return Err(format!("len(x) ({}) != mat.cols ({})", x.len(), self.cols));
        }
        let mut result = vec![T::zero(); self.rows];
        csc_matvec(
            self.rows,
            self.cols,
            &self.colptr,
            &self.rowidx,
            &self.data,
            x,
            &mut result,
        );
        Ok(result)
    }

    /// Performs matrix-matrix multiplication. The number of rows of x
    /// must equal the number of columns of the receiver.
    pub fn mat_mat(&self, x: &CSC<I, T>) -> Result<CSC<I, T>, String> {
        if x.rows != self.cols {
            return Err(format!(
                "dimension mismatch, rows {} cols {}",
                x.rows, self.cols
            ));
        }
        let rows = self.rows;
        let cols = x.cols;

        // let nnz = csr_matmat_maxnnz(cols, rows, &x.colptr, &x.rowidx, &self.colptr, &self.rowidx);
        let nnz = csc_matmat_maxnnz(rows, cols, &self.colptr, &self.rowidx, &x.colptr, &x.rowidx);

        // let nnz = colptr[cols];

        let mut colptr = vec![I::zero(); cols + 1];
        let mut rowidx = vec![I::zero(); nnz];
        let mut data = vec![T::zero(); nnz];

        // csr_matmat(cols, rows, &x.colptr, &x.rowidx, &x.data, &self.colptr, &self.rowidx, &self.data, &mut colptr, &mut rowidx, &mut data);
        csc_matmat(
            rows,
            cols,
            &self.colptr,
            &self.rowidx,
            &self.data,
            &x.colptr,
            &x.rowidx,
            &x.data,
            &mut colptr,
            &mut rowidx,
            &mut data,
        );

        CSC::new(rows, cols, rowidx, colptr, data)
    }

    /// Creates a new CSC matrix with only the selected rows and columns of the receiver.
    pub fn select(&self, rowidx: &[I], colidx: &[I]) -> Result<CSC<I, T>, String> {
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
            self.cols,
            self.rows,
            &self.colptr,
            &self.rowidx,
            &self.data,
            &colidx,
            &rowidx,
            &mut Bp,
            &mut Bj,
            &mut Bx,
        );

        Ok(CSC {
            rows: rowidx.len(),
            cols: colidx.len(),
            rowidx: Bj,
            colptr: Bp,
            data: Bx,
        })
    }

    /// Performs matrix-matrix addition. The dimensions of x must be equal to those of the receiver.
    pub fn add(&self, x: &CSC<I, T>) -> Result<CSC<I, T>, String> {
        if x.rows != self.rows || x.cols != self.cols {
            return Err(format!(
                "dimension mismatch ({},{}) != ({},{})",
                x.rows, x.cols, self.rows, self.cols
            ));
        }

        let maxnnz = self.nnz() + x.nnz();
        let mut rowidx = vec![I::zero(); maxnnz];
        let mut colptr = vec![I::zero(); self.colptr.len()];
        let mut data = vec![T::zero(); maxnnz];

        // fixme
        // csr_plus_csr(
        //     self.cols,
        //     self.rows,
        //     &self.colptr,
        //     &self.rowidx,
        //     &self.data,
        //     &x.colptr,
        //     &x.rowidx,
        //     &x.data,
        //     &mut colptr,
        //     &mut rowidx,
        //     &mut data,
        // );

        let newnnz = colptr[self.cols];
        let rowidx = rowidx[..newnnz];
        let data = data[..newnnz];

        CSC::new(self.rows, self.cols, rowidx, colptr, data)
    }

    /// Performs matrix-matrix subtraction. The dimensions of x must be
    /// equal to those of the receiver.
    pub fn subtract(&self, x: &CSC<I, T>) -> Result<CSC<I, T>, String> {
        if x.rows != self.rows || x.cols != self.cols {
            return Err(format!(
                "dimension mismatch ({},{}) != ({},{})",
                x.rows, x.cols, self.rows, self.cols
            ));
        }

        let maxnnz = self.nnz() + x.nnz();
        let mut rowidx = vec![I::zero(); maxnnz];
        let mut colptr = vec![I::zero(); self.colptr.len()];
        let mut data = vec![T::zero(); maxnnz];
        // fixme
        // csr_minus_csr(
        //     self.cols,
        //     self.rows,
        //     &self.colptr,
        //     &self.rowidx,
        //     &self.data,
        //     &x.colptr,
        //     &x.rowidx,
        //     &x.data,
        //     &mut colptr,
        //     &mut rowidx,
        //     &mut data,
        // );

        let newnnz = colptr[self.cols];
        let rowidx = rowidx[..newnnz];
        let data = data[..newnnz];

        CSC::new(self.rows, self.cols, rowidx, colptr, data)
    }

    /// Converts the matrix into Compressed Sparse Row (CSR) format.
    pub fn to_csr(&self) -> CSR<I, T> {
        let mut rowptr = vec![I::zero(); self.rows + 1];
        let mut colidx = vec![I::zero(); self.nnz()];
        let mut data = vec![T::zero(); self.nnz()];

        // csr_tocsc(self.cols, self.rows, &self.colptr, &self.rowidx, &self.data, &mut rowptr, &mut colidx, &mut data); FIXME
        csc_tocsr(
            self.rows,
            self.cols,
            &self.rowidx,
            &self.colptr,
            &self.data,
            &mut colidx,
            &mut rowptr,
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

    /// Converts the matrix into Coordinate (Coo) format.
    pub fn to_coo(&self) -> Coo<I, T> {
        let mut rowidx = vec![I::zero(); self.nnz()];
        let mut colidx = vec![I::zero(); self.nnz()];
        let mut data = vec![T::zero(); self.nnz()];

        csr_tocoo(
            self.cols,
            self.rows,
            &self.colptr,
            &self.rowidx,
            &self.data,
            &mut colidx,
            &mut rowidx,
            &mut data,
        );

        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colidx,
            data,
        }
    }

    /// Converts the matrix into a dense 2D slice.
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        self.to_csr().to_dense()
    }

    /// Returns a slice of the elements on the main diagonal.
    pub fn diagonal(&self) -> Vec<T> {
        let mut diag = vec![T::zero(); min(self.rows, self.cols)];
        csr_diagonal(
            self.cols,
            self.rows,
            &self.colptr,
            &self.rowidx,
            &self.data,
            &mut diag,
        );
        diag
    }

    /// Assembles a matrix from four sub-matrices using HStackCoo and VStackCoo.
    ///        -----------
    ///       | J11 | J12 |
    ///   J = |-----------|
    ///       | J21 | J22 |
    ///        -----------
    /// An error is returned if the matrix dimensions do not match correctly.
    pub fn compose(
        J11: &CSC<I, T>,
        J12: &CSC<I, T>,
        J21: &CSC<I, T>,
        J22: &CSC<I, T>,
    ) -> Result<Self, String> {
        let J11coo = J11.to_coo();
        let J12coo = J12.to_coo();
        let J21coo = J21.to_coo();
        let J22coo = J22.to_coo();

        let J1X = Coo::h_stack(&J11coo, &J12coo)?;
        let J2X = Coo::h_stack(&J21coo, &J22coo)?;

        let J = Coo::v_stack(&J1X, &J2X)?;

        Ok(J.to_csc())
    }
}
