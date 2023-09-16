use crate::col::{csc_matmat, csc_matmat_maxnnz, csc_matvec};
use crate::coo::Coo;
use crate::csr::CSR;
use crate::csr_tocsc;
use crate::row::{
    csr_add_csr, csr_diagonal, csr_has_canonical_format, csr_has_sorted_indices, csr_select,
    csr_sort_indices, csr_sub_csr, csr_sum_duplicates, csr_tocoo,
};
use crate::traits::{Integer, Scalar};
use anyhow::{format_err, Result};
use std::cmp::min;

/// A sparse matrix with scalar values stored in Compressed Sparse Column (CSC) format.
pub struct CSC<I, T> {
    rows: usize,
    cols: usize,
    pub(super) rowidx: Vec<I>,
    pub(super) colptr: Vec<I>,
    pub(super) data: Vec<T>,
}

impl<I: Integer, T: Scalar> CSC<I, T> {
    /// Creates a new CSC matrix. Inputs are not copied. An error
    /// is returned if the slice arguments do not have the correct length.
    pub fn new(
        rows: usize,
        cols: usize,
        rowidx: Vec<I>,
        colptr: Vec<I>,
        data: Vec<T>,
    ) -> Result<Self> {
        if colptr.len() != cols + 1 {
            return Err(format_err!("colptr has invalid length"));
        }
        let nnz = colptr[colptr.len() - 1].to_usize().unwrap();
        if rowidx.len() < nnz {
            return Err(format_err!("rowidx array has fewer than nnz elements"));
        }
        if data.len() < nnz {
            return Err(format_err!("data array has fewer than nnz elements"));
        }
        Ok(Self {
            rows,
            cols,
            rowidx,
            colptr,
            data,
        })
    }

    pub fn from_dense(a: &[Vec<T>]) -> Self {
        let rows = a.len();
        let cols = a[0].len();

        let mut colptr = Vec::<I>::new();
        let mut rowidx = Vec::<I>::new();
        let mut data = Vec::new();

        let mut idxptr: usize = 0;
        for i in 0..cols {
            colptr.push(I::from(idxptr).unwrap());
            for j in 0..rows {
                if a[j][i] != T::zero() {
                    data.push(a[j][i]);
                    rowidx.push(I::from(j).unwrap());
                    idxptr += 1
                }
            }
        }
        colptr.push(I::from(idxptr).unwrap());
        let mut csc = CSC {
            rows,
            cols,
            rowidx,
            colptr,
            data,
        };
        csc.prune().unwrap();
        csc
    }

    /// Creates a new CSC matrix with the given values on the main
    /// diagonal. The input is not copied.
    pub fn with_diagonal(data: Vec<T>) -> Self {
        let n = data.len();
        let mut rowidx = vec![I::zero(); n];
        for i in 0..rowidx.len() {
            rowidx[i] = I::from(i).unwrap();
        }
        let mut colptr = vec![I::zero(); n + 1];
        for i in 0..colptr.len() {
            colptr[i] = I::from(i).unwrap();
        }
        Self {
            rows: n,
            cols: n,
            rowidx,
            colptr,
            data,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row indexes (size nnz).
    pub fn rowidx(&self) -> &[I] {
        &self.rowidx
    }

    /// Column pointers (size cols+1).
    pub fn colptr(&self) -> &[I] {
        &self.colptr
    }

    /// Explicitly stored values (size nnz).
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the count of explicitly stored values (nonzeros).
    pub fn nnz(&self) -> usize {
        if self.colptr.len() == 0 {
            0
        } else {
            self.colptr[self.colptr.len() - 1].to_usize().unwrap()
        }
    }

    //     /// Copy creates an identical CSC matrix with newly allocated
    //     /// index, pointer and data slices.
    //     pub fn copy(&self) -> CSC<I, T> {
    //         // let rowidx = vec!(I::zero(); self.rowidx.len());
    //         // let colptr = vec!(I::zero(); self.colptr.len());
    //         // let data = vec!(T::zero(); self.data.len());
    //         // copy(rowidx, self.rowidx);
    //         // copy(colptr, self.colptr);
    //         // copy(data, self.data);
    //
    //         CSC {
    //             rows: self.rows,
    //             cols: self.cols,
    //             rowidx: self.rowidx.clone(),
    //             colptr: self.colptr.clone(),
    //             data: self.data.clone(),
    //         }
    //     }

    /// Transpose creates a CSR matrix that is the transpose of the
    /// receiver.
    pub fn transpose(&self) -> CSR<I, T> {
        CSR::new(
            self.cols,
            self.rows,
            self.colptr.clone(),
            self.rowidx.clone(),
            self.data.clone(),
        )
        .unwrap()
    }

    /// An alias for Transpose.
    pub fn t(&self) -> CSR<I, T> {
        self.transpose()
    }

    /// Returns true if the matrix has sorted indexes and no duplicates.
    pub fn has_canonical_format(&self) -> bool {
        csr_has_canonical_format::<I>(self.colptr.len() - 1, &self.colptr, &self.rowidx)
    }

    /// Returns true if the indexes of the matrix are in sorted order.
    pub fn has_sorted_indexes(&self) -> bool {
        csr_has_sorted_indices::<I>(self.colptr.len() - 1, &self.colptr, &self.rowidx)
    }

    /// Sorts the indexes of this matrix in place.
    pub fn sort_indexes(&mut self) {
        csr_sort_indices::<I, T>(
            self.colptr.len() - 1,
            &self.colptr,
            &mut self.rowidx,
            &mut self.data,
        )
    }

    /// Sums duplicate entries.
    pub fn sum_duplicates(&mut self) {
        if self.has_canonical_format() {
            return;
        }
        self.sort_indexes();

        csr_sum_duplicates(
            self.cols,
            self.rows,
            &mut self.colptr,
            &mut self.rowidx,
            &mut self.data,
        );

        self.prune().unwrap(); // nnz may have changed
    }

    /// Removes empty space after all non-zero elements.
    pub fn prune(&mut self) -> Result<()> {
        if self.colptr.len() != self.cols + 1 {
            return Err(format_err!("index pointer has invalid length"));
        }
        let nnz = self.nnz();
        if self.rowidx.len() < nnz {
            return Err(format_err!("indexes array has fewer than nnz elements"));
        }
        if self.data.len() < nnz {
            return Err(format_err!("data array has fewer than nnz elements"));
        }

        self.data = self.data[..nnz].to_owned();
        self.rowidx = self.rowidx[..nnz].to_owned();
        Ok(())
    }

    /// Performs matrix-vector multiplication. The length of x must be equal
    /// to the number of columns of the receiver.
    pub fn mat_vec(&self, x: &[T]) -> Result<Vec<T>> {
        if x.len() != self.cols {
            return Err(format_err!(
                "len(x) ({}) != mat.cols ({})",
                x.len(),
                self.cols
            ));
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
    pub fn mat_mat(&self, x: &CSC<I, T>) -> Result<CSC<I, T>> {
        if x.rows != self.cols {
            return Err(format_err!(
                "dimension mismatch, rows {} cols {}",
                x.rows,
                self.cols
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
    pub fn select(&self, rowidx: Option<&[I]>, colidx: Option<&[I]>) -> Result<CSC<I, T>> {
        if !self.has_canonical_format() {
            // self.sum_duplicates();
            return Err(format_err!("must have canonical form, sum_duplicates"));
        }

        let mut rowidx_full = Vec::new();
        let rowidx = match rowidx {
            None => {
                rowidx_full.resize(self.rows, I::zero());
                for i in 0..rowidx_full.len() {
                    rowidx_full[i] = I::from(i).unwrap();
                }
                rowidx_full.as_slice()
            }
            Some(rowidx) => {
                for &ri in rowidx {
                    if ri < I::zero() || ri >= I::from(self.rows).unwrap() {
                        return Err(format_err!("out of range: {}", ri));
                    }
                }
                rowidx
            }
        };

        let mut colidx_full = Vec::new();
        let colidx = match colidx {
            None => {
                colidx_full.resize(self.cols, I::zero());
                for i in 0..colidx_full.len() {
                    colidx_full[i] = I::from(i).unwrap();
                }
                colidx_full.as_slice()
            }
            Some(colidx) => {
                for &ci in colidx {
                    if ci < I::zero() || ci >= I::from(self.cols).unwrap() {
                        return Err(format_err!("out of range: {}", ci));
                    }
                }
                colidx
            }
        };

        let mut b_p = Vec::<I>::new();
        let mut b_j = Vec::<I>::new();
        let mut b_x = Vec::<T>::new();

        csr_select(
            self.cols,
            self.rows,
            &self.colptr,
            &self.rowidx,
            &self.data,
            colidx,
            rowidx,
            &mut b_p,
            &mut b_j,
            &mut b_x,
        );

        Ok(CSC {
            rows: rowidx.len(),
            cols: colidx.len(),
            rowidx: b_j,
            colptr: b_p,
            data: b_x,
        })
    }

    /// Performs matrix-matrix addition. The dimensions of x must be equal to those of the receiver.
    pub fn add(&self, x: &CSC<I, T>) -> Result<CSC<I, T>> {
        if x.rows != self.rows || x.cols != self.cols {
            return Err(format_err!(
                "dimension mismatch ({},{}) != ({},{})",
                x.rows,
                x.cols,
                self.rows,
                self.cols
            ));
        }

        let maxnnz = self.nnz() + x.nnz();
        let mut rowidx = vec![I::zero(); maxnnz];
        let mut colptr = vec![I::zero(); self.colptr.len()];
        let mut data = vec![T::zero(); maxnnz];

        csr_add_csr(
            self.cols,
            self.rows,
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

        let newnnz = colptr[self.cols].to_usize().unwrap();
        let rowidx = &rowidx[..newnnz];
        let data = &data[..newnnz];

        CSC::new(
            self.rows,
            self.cols,
            rowidx.to_owned(),
            colptr,
            data.to_owned(),
        )
    }

    /// Performs matrix-matrix subtraction. The dimensions of x must be
    /// equal to those of the receiver.
    pub fn subtract(&self, x: &CSC<I, T>) -> Result<CSC<I, T>> {
        if x.rows != self.rows || x.cols != self.cols {
            return Err(format_err!(
                "dimension mismatch ({},{}) != ({},{})",
                x.rows,
                x.cols,
                self.rows,
                self.cols
            ));
        }

        let maxnnz = self.nnz() + x.nnz();
        let mut rowidx = vec![I::zero(); maxnnz];
        let mut colptr = vec![I::zero(); self.colptr.len()];
        let mut data = vec![T::zero(); maxnnz];

        csr_sub_csr(
            self.cols,
            self.rows,
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

        let newnnz = colptr[self.cols].to_usize().unwrap();
        let rowidx = &rowidx[..newnnz];
        let data = &data[..newnnz];

        CSC::new(
            self.rows,
            self.cols,
            rowidx.to_owned(),
            colptr,
            data.to_owned(),
        )
    }

    /// Converts the matrix into Compressed Sparse Row (CSR) format.
    pub fn to_csr(&self) -> CSR<I, T> {
        let mut rowptr = vec![I::zero(); self.rows + 1];
        let mut colidx = vec![I::zero(); self.nnz()];
        let mut data = vec![T::zero(); self.nnz()];

        csr_tocsc(
            self.cols,
            self.rows,
            &self.colptr,
            &self.rowidx,
            &self.data,
            &mut rowptr,
            &mut colidx,
            &mut data,
        );

        // csc_tocsr(
        //     self.rows,
        //     self.cols,
        //     &self.rowidx,
        //     &self.colptr,
        //     &self.data,
        //     &mut colidx,
        //     &mut rowptr,
        //     &mut data,
        // );

        CSR::new(self.rows, self.cols, rowptr, colidx, data).unwrap()
    }

    /// Converts the matrix into Coordinate (Coo) format.
    pub fn to_coo(&self) -> Coo<I, T> {
        let nnz = self.nnz();
        let mut rowidx = vec![I::zero(); nnz];
        let mut colidx = vec![I::zero(); nnz];
        let mut data = vec![T::zero(); nnz];

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

        Coo::new(self.rows, self.cols, rowidx, colidx, data).unwrap()
    }

    /// Converts the matrix into a dense 2D slice.
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        self.to_csr().to_dense()
    }

    /// Returns a slice of the elements on the main diagonal.
    pub fn diagonal(&self) -> Vec<T> {
        let size = min(self.rows, self.cols);
        let mut diag = vec![T::zero(); size];
        csr_diagonal(
            0,
            self.cols,
            self.rows,
            &self.colptr,
            &self.rowidx,
            &self.data,
            &mut diag,
        );
        diag
    }

    pub fn to_string(&self) -> String {
        let mut buf: String = String::new();
        for j in 0..self.cols {
            let col_start = self.colptr[j].to_usize().unwrap();
            let col_end = self.colptr[j + 1].to_usize().unwrap();

            for ii in col_start..col_end {
                let i = self.rowidx[ii].to_usize().unwrap();

                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(&format!("({}, {}) {}", i, j, self.data[ii]));
            }
        }
        buf
    }

    /// Assembles a matrix from four sub-matrices using HStackCoo and VStackCoo.
    ///        -----------
    ///       | J11 | J12 |
    ///   J = |-----------|
    ///       | J21 | J22 |
    ///        -----------
    /// An error is returned if the matrix dimensions do not match correctly.
    pub fn compose(
        j11: &CSC<I, T>,
        j12: &CSC<I, T>,
        j21: &CSC<I, T>,
        j22: &CSC<I, T>,
    ) -> Result<Self> {
        let j11coo = j11.to_coo();
        let j12coo = j12.to_coo();
        let j21coo = j21.to_coo();
        let j22coo = j22.to_coo();

        let j1x = Coo::h_stack(&j11coo, &j12coo)?;
        let j2x = Coo::h_stack(&j21coo, &j22coo)?;

        let j = Coo::v_stack(&j1x, &j2x)?;

        Ok(j.to_csc())
    }
}
