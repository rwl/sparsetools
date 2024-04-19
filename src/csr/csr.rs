use crate::coo::Coo;
use crate::csc::CSC;
use crate::csr::table::csr_table;
use crate::graph::cs_graph_components;
use crate::row::{
    csr_diagonal, csr_has_canonical_format, csr_has_sorted_indices, csr_matmat, csr_matmat_maxnnz,
    csr_matvec, csr_select, csr_sort_indices, csr_sum_duplicates, csr_tocsc, csr_todense,
    expandptr,
};
use crate::traits::{Integer, Scalar};
use anyhow::{format_err, Result};
use std::cmp::min;

/// A sparse matrix with scalar values stored in Compressed Sparse Row (CSR) format.
#[derive(Clone)]
pub struct CSR<I, T> {
    rows: usize,
    cols: usize,
    pub(super) rowptr: Vec<I>,
    pub(super) colidx: Vec<I>,
    pub(super) values: Vec<T>,
}

impl<I: Integer, T: Scalar> CSR<I, T> {
    /// Creates a new CSR matrix. Inputs are not copied. An error
    /// is returned if the slice arguments do not have the correct length.
    pub fn new(
        rows: usize,
        cols: usize,
        rowptr: Vec<I>,
        colidx: Vec<I>,
        values: Vec<T>,
    ) -> Result<Self> {
        if rowptr.len() != rows + 1 {
            return Err(format_err!("rowptr has invalid length"));
        }
        let nnz = rowptr[rowptr.len() - 1].to_usize().unwrap();
        if colidx.len() < nnz {
            return Err(format_err!("colidx has fewer than nnz elements"));
        }
        if values.len() < nnz {
            return Err(format_err!("values array has fewer than nnz elements"));
        }
        Ok(CSR {
            rows,
            cols,
            rowptr,
            colidx,
            values,
        })
    }

    pub fn with_size(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            rowptr: vec![I::zero(); rows + 1],
            colidx: vec![I::zero(); 0],
            values: vec![T::zero(); 0],
        }
    }

    pub fn from_dense(a: &[Vec<T>]) -> Self {
        let rows = a.len();
        let cols = a[0].len();

        let mut rowptr = Vec::<I>::new();
        let mut colidx = Vec::<I>::new();
        let mut values = Vec::new();

        let mut idxptr: usize = 0;
        for i in 0..rows {
            rowptr.push(I::from(idxptr).unwrap());
            for j in 0..cols {
                if a[i][j] != T::zero() {
                    values.push(a[i][j]);
                    colidx.push(I::from(j).unwrap());
                    idxptr += 1
                }
            }
        }
        rowptr.push(I::from(idxptr).unwrap());
        let mut csr = CSR {
            rows,
            cols,
            rowptr,
            colidx,
            values,
        };
        csr.prune().unwrap();
        csr
    }

    /// Creates a new CSR matrix with the given values on the main
    /// diagonal. The input is not copied.
    pub fn with_diagonal(values: Vec<T>) -> Self {
        let n = values.len();
        let mut rowptr = vec![I::zero(); n + 1];
        for i in 0..n + 1 {
            rowptr[i] = I::from(i).unwrap();
        }
        let mut colidx = vec![I::zero(); n];
        for i in 0..n {
            colidx[i] = I::from(i).unwrap();
        }
        CSR {
            rows: n,
            cols: n,
            rowptr,
            colidx,
            values,
        }
    }

    pub fn h_stack(a_mat: &CSR<I, T>, b_mat: &CSR<I, T>) -> Result<CSR<I, T>> {
        Ok(Coo::h_stack(&a_mat.to_coo(), &b_mat.to_coo())?.to_csr())
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row pointers (size rows+1).
    pub fn rowptr(&self) -> &[I] {
        &self.rowptr
    }

    /// Column indexes (size nnz).
    pub fn colidx(&self) -> &[I] {
        &self.colidx
    }

    /// Explicitly stored values (size nnz).
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Explicitly stored values (size nnz).
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Returns the count of explicitly-stored values (nonzeros).
    pub fn nnz(&self) -> usize {
        if self.rowptr.is_empty() {
            0
        } else {
            self.rowptr[self.rowptr.len() - 1].to_usize().unwrap()
        }
    }

    /// Creates a CSC matrix that is the transpose of the receiver.
    /// The underlying index and values slices are not copied.
    pub fn transpose(self) -> CSC<I, T> {
        CSC::new(self.cols, self.rows, self.colidx, self.rowptr, self.values).unwrap()
    }

    /// Returns the transpose of the receiver.
    /// The underlying index and values slices **are** copied.
    pub fn t(&self) -> CSC<I, T> {
        CSC::new(
            self.cols,
            self.rows,
            self.colidx.clone(),
            self.rowptr.clone(),
            self.values.clone(),
        )
        .unwrap()
    }

    /// Returns true if the matrix has sorted indexes and no duplicates.
    pub fn has_canonical_format(&self) -> bool {
        csr_has_canonical_format::<I>(self.rowptr.len() - 1, &self.rowptr, &self.colidx)
    }

    /// HasSortedIndexes returns true if the indexes of the matrix are in
    /// sorted order.
    pub fn has_sorted_indexes(&self) -> bool {
        csr_has_sorted_indices::<I>(self.rowptr.len() - 1, &self.rowptr, &self.colidx)
    }

    /// SortIndexes sorts the indexes of this matrix in place.
    pub fn sort_indexes(&mut self) {
        csr_sort_indices::<I, T>(
            self.rowptr.len() - 1,
            &self.rowptr,
            &mut self.colidx,
            &mut self.values,
        )
    }

    /// Sums duplicate entries.
    pub fn sum_duplicates(&mut self) -> Result<()> {
        if self.has_canonical_format() {
            return Ok(());
        }
        self.sort_indexes();

        csr_sum_duplicates(
            self.rows,
            self.cols,
            &mut self.rowptr,
            &mut self.colidx,
            &mut self.values,
        );

        self.prune() // nnz may have changed
    }

    // Prune removes empty space after all non-zero elements.
    fn prune(&mut self) -> Result<()> {
        if self.rowptr.len() != self.rows + 1 {
            return Err(format_err!("index pointer has invalid length"));
        }
        let nnz = self.nnz();
        if self.colidx.len() < nnz {
            return Err(format_err!("indexes array has fewer than nnz elements"));
        }
        if self.values.len() < nnz {
            return Err(format_err!("values array has fewer than nnz elements"));
        }

        // self.values = self.values.[..nnz];
        self.values.resize(nnz, T::zero());
        // self.colidx = self.colidx[..nnz];
        self.colidx.resize(nnz, I::zero());

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
        csr_matvec(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.values,
            x,
            &mut result,
        );
        Ok(result)
    }

    /// Performs matrix-matrix multiplication. The number of rows of x
    /// must equal the number of columns of the receiver.
    pub fn mat_mat(&self, x: &CSR<I, T>) -> Result<CSR<I, T>> {
        if self.cols != x.rows {
            return Err(format_err!(
                "dimension mismatch, rows {} cols {}",
                x.rows,
                self.cols
            ));
        }
        let rows = self.rows;
        let cols = x.cols;

        let mut rowptr = vec![I::zero(); rows + 1];

        let nnz = csr_matmat_maxnnz(rows, cols, &self.rowptr, &self.colidx, &x.rowptr, &x.colidx); //, &rowptr);

        let mut colidx = vec![I::zero(); nnz];
        let mut values = vec![T::zero(); nnz];

        csr_matmat(
            rows,
            cols,
            &self.rowptr,
            &self.colidx,
            &self.values,
            &x.rowptr,
            &x.colidx,
            &x.values,
            &mut rowptr,
            &mut colidx,
            &mut values,
        );

        CSR::new(rows, cols, rowptr, colidx, values)
    }

    /// Creates a new CSR matrix with only the selected rows and columns
    /// of the receiver.
    pub fn select(&self, rowidx: Option<&[I]>, colidx: Option<&[I]>) -> Result<CSR<I, T>> {
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

        let mut b_p: Vec<I> = vec![];
        let mut b_j: Vec<I> = vec![];
        let mut b_x: Vec<T> = vec![];

        csr_select(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.values,
            &rowidx,
            &colidx,
            &mut b_p,
            &mut b_j,
            &mut b_x,
        );

        CSR::new(rowidx.len(), colidx.len(), b_p, b_j, b_x)
    }

    /// Converts the matrix into Coordinate (Coo) format.
    pub fn to_coo(&self) -> Coo<I, T> {
        // let values = vec![T::zero(); self.values.len()];
        // copy(values, self.values);
        let values = self.values.clone();

        // let colidx = vec![I::zero(); self.colidx.len()];
        // copy(colidx, self.colidx);
        let colidx = self.colidx.clone();

        let mut rowidx = vec![I::zero(); self.colidx.len()];

        expandptr(self.rows, &self.rowptr, &mut rowidx);

        Coo::new(self.rows, self.cols, rowidx, colidx, values).unwrap()
    }

    /// Converts the matrix to Compressed Sparse Column (CSC) format.
    pub fn to_csc(&self) -> CSC<I, T> {
        let mut colptr = vec![I::zero(); self.cols + 1];
        let mut rowidx = vec![I::zero(); self.nnz()];
        let mut values = vec![T::zero(); self.nnz()];

        csr_tocsc(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.values,
            &mut colptr,
            &mut rowidx,
            &mut values,
        );

        CSC::new(self.rows, self.cols, rowidx, colptr, values).unwrap()
    }

    /// Converts the matrix into a dense 2D slice.
    pub(crate) fn to_dense(&self) -> Vec<Vec<T>> {
        // let mut dense = vec![&mut vec![T::zero(); cols]; rows];
        let mut dense = vec![vec![T::zero(); self.cols]; self.rows];
        // for r := range dense {
        // 	dense[r] = make([]float64, self.cols)
        // }
        csr_todense(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.values,
            &mut dense,
        );
        dense
    }

    /// Returns a vector of the elements on the main diagonal.
    pub fn diagonal(&self) -> Vec<T> {
        let size = min(self.rows, self.cols);
        let mut diag = vec![T::zero(); size];
        csr_diagonal(
            0,
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.values,
            &mut diag,
        );
        diag
    }

    /// Determine connected compoments of a compressed sparse graph.
    pub fn connected_components<F: Integer + num_traits::Signed>(&self) -> Result<(I, Vec<F>)> {
        if self.cols != self.rows {
            return Err(format_err!(
                "matrix must be square, rows = {} cols = {}",
                self.rows,
                self.cols
            ));
        }
        let n = self.rows;
        let mut flag = vec![F::zero(); n];
        let ncc = cs_graph_components::<I, T, F>(n, &self.rowptr, &self.colidx, &mut flag)?;
        Ok((ncc, flag))
    }

    /// Returns a string representation of the matrix.
    pub fn to_string(&self) -> String {
        // let w = csr_string(
        //     self.rows,
        //     self.cols,
        //     &self.rowptr,
        //     &self.colidx,
        //     &self.values,
        //     vec![],
        // );
        use std::io::Write;

        let mut w = Vec::default();
        for i in 0..self.rows {
            let start = self.rowptr[i].to_usize().unwrap();
            let end = self.rowptr[i + 1].to_usize().unwrap();

            for jj in start..end {
                writeln!(w, "({}, {}) {}", i, self.colidx[jj], self.values[jj]).unwrap();
            }
        }

        String::from_utf8(w).unwrap()
    }

    /// Returns a tabular representation of the matrix.
    pub fn to_table(&self) -> String {
        let w = csr_table(
            self.rows,
            self.cols,
            &self.rowptr,
            &self.colidx,
            &self.values,
            false,
            vec![],
            None,
        );
        String::from_utf8(w)
            .unwrap()
            .trim_end_matches("\n")
            .to_string()
    }
}
