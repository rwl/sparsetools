use crate::coord::coo_tocsr;
use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use anyhow::{format_err, Result};
use std::cmp::min;

/// A sparse matrix with scalar values stored in Coordinate format
/// (also called "aij", "ijv" or "triplet" format).
#[derive(Clone)]
pub struct Coo<I, T> {
    rows: usize,
    cols: usize,
    pub(super) rowidx: Vec<I>,
    pub(super) colidx: Vec<I>,
    pub(super) values: Vec<T>,
}

impl<I: Integer, T: Scalar> Coo<I, T> {
    /// Creates a new coordinate matrix. Inputs are not copied. An error
    /// is returned if the slice arguments do not have the same length.
    pub fn new(
        rows: usize,
        cols: usize,
        rowidx: Vec<I>,
        colidx: Vec<I>,
        values: Vec<T>,
    ) -> Result<Self> {
        let nnz = values.len();
        if nnz != rowidx.len() || nnz != colidx.len() {
            return Err(format_err!(
                "row, column, and values array must all be the same length"
            ));
        }
        Ok(Self {
            rows,
            cols,
            rowidx,
            colidx,
            values,
        })
    }

    pub fn with_size(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            rowidx: Vec::default(),
            colidx: Vec::default(),
            values: Vec::default(),
        }
    }

    pub fn with_capacity(rows: usize, cols: usize, nnz: usize) -> Self {
        Self {
            rows,
            cols,
            rowidx: Vec::with_capacity(nnz),
            colidx: Vec::with_capacity(nnz),
            values: Vec::with_capacity(nnz),
        }
    }

    /// Creates a sparse matrix in coordinate format from a full matrix.
    pub fn from_dense(a: &[Vec<T>]) -> Self {
        let (m, mut n) = (a.len(), 0);
        if m > 0 {
            n = a[0].len();
        }

        let mut rowidx = Vec::<I>::with_capacity(m * n);
        let mut colidx = Vec::<I>::with_capacity(m * n);
        let mut values = Vec::<T>::with_capacity(m * n);

        for (i, r) in a.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                if v != T::zero() {
                    rowidx.push(I::from(i).unwrap());
                    colidx.push(I::from(j).unwrap());
                    values.push(v);
                }
            }
        }

        Self {
            rows: m,
            cols: n,
            rowidx,
            colidx,
            values,
        }
    }

    /// Diagonal creates a coordinate matrix with the given diagonal.
    pub fn with_diagonal(diag: &[T]) -> Self {
        let n = diag.len();
        let mut rowidx = vec![I::zero(); n];
        let mut colidx = vec![I::zero(); n];
        let mut values = vec![T::zero(); n];
        for i in 0..n {
            rowidx[i] = I::from(i).unwrap();
            colidx[i] = I::from(i).unwrap();
            values[i] = diag[i];
        }
        Self {
            rows: n,
            cols: n,
            rowidx,
            colidx,
            values,
        }
    }

    // /// Generates a coordinate matrix with the given size and
    // /// density with randomly distributed values.
    // pub fn random(rows: I, cols: I, density: f64) -> Self {
    //     let mut rng = rand::thread_rng();
    //
    //     let density = match density {
    //         _ if density < 0.0 => 0.0,
    //         _ if density > 1.0 => 1.0,
    //         _ => density,
    //     };
    //
    //     let size = (rows * cols).to_usize().unwrap();
    //
    //     let nnz = (density * (rows * cols).to_f64().unwrap()) as usize;
    //
    //     let mut idx = vec![I::zero(); nnz];
    //     let mut selected = HashSet::<usize>::new();
    //     for i in 0..nnz {
    //         let mut j = rng.gen_range(0..size);
    //         loop {
    //             if !selected.contains(&j) {
    //                 break;
    //             }
    //             j = rng.gen_range(0..size);
    //         }
    //         selected.insert(j);
    //         idx[i] = I::from(j).unwrap();
    //     }
    //
    //     let mut colidx = vec![I::zero(); nnz];
    //     let mut rowidx = vec![I::zero(); nnz];
    //     for (i, &v) in idx.iter().enumerate() {
    //         let c = (v.to_f64().unwrap() * 1.0 / rows.to_f64().unwrap()).floor();
    //         colidx[i] = I::from(c).unwrap();
    //         rowidx[i] = v - colidx[i] * rows;
    //     }
    //     let mut values = vec![T::zero(); nnz];
    //     for i in 0..values.len() {
    //         values[i] = rng.gen();
    //     }
    //
    //     Self {
    //         rows,
    //         cols,
    //         rowidx,
    //         colidx,
    //         values,
    //     }
    // }

    pub fn identity(n: usize) -> Self {
        let mut rowidx = vec![I::zero(); n];
        let mut colidx = vec![I::zero(); n];
        let mut values = vec![T::zero(); n];
        for i in 0..n {
            rowidx[i] = I::from(i).unwrap();
            colidx[i] = I::from(i).unwrap();
            values[i] = T::one();
        }
        Self {
            rows: n,
            cols: n,
            rowidx,
            colidx,
            values,
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

    /// Column indexes (size nnz).
    pub fn colidx(&self) -> &[I] {
        &self.colidx
    }

    /// Explicitly stored values (size nnz).
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the count of explicitly stored values (nonzeros).
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn push(&mut self, row: I, col: I, v: T) {
        self.rowidx.push(row);
        self.colidx.push(col);
        self.values.push(v);
    }

    pub fn extend(&mut self, row: &[I], col: &[I], v: &[T]) {
        self.rowidx.extend(row);
        self.colidx.extend(col);
        self.values.extend(v);
    }

    // Copy creates an identical coordinate matrix with newly allocated
    // index and values slices.
    // func (mat *Coo) Copy() *Coo {
    // 	nnz := mat.NNZ()
    //
    // 	rowidx := make([]int, nnz)
    // 	colidx := make([]int, nnz)
    // 	values := make([]float64, nnz)
    // 	copy(rowidx, mat.rowidx)
    // 	copy(colidx, mat.colidx)
    // 	copy(values, mat.values)
    //
    // 	return &Coo{rows: mat.rows, cols: mat.cols,
    // 		rowidx: rowidx, colidx: colidx, values: values}
    // }

    /// Creates a coordinate matrix that is the transpose of the
    /// receiver. The underlying index and values vectors are not copied.
    pub fn transpose(self) -> Coo<I, T> {
        Coo {
            rows: self.cols,
            cols: self.rows,
            rowidx: self.colidx,
            colidx: self.rowidx,
            values: self.values,
        }
    }

    /// An alias for `transpose`, but the underlying index and values
    /// vectors are copied.
    pub fn t(&self) -> Coo<I, T> {
        Coo {
            rows: self.cols,
            cols: self.rows,
            rowidx: self.colidx.clone(),
            colidx: self.rowidx.clone(),
            values: self.values.clone(),
        }
    }

    /// Converts the matrix to Compressed Sparse Column (CSC) format.
    /// Duplicate entries are summed.
    pub fn to_csc(&self) -> CSC<I, T> {
        let nnz = self.nnz();
        let mut rowidx = vec![I::zero(); nnz];
        let mut colptr = vec![I::zero(); self.cols + 1];
        let mut values = vec![T::zero(); nnz];

        coo_tocsr(
            self.cols,
            self.rows,
            self.nnz(),
            &self.colidx,
            &self.rowidx,
            &self.values,
            &mut colptr,
            &mut rowidx,
            &mut values,
        );

        let mut a_mat = CSC::new(self.rows, self.cols, rowidx, colptr, values).unwrap();
        a_mat.sum_duplicates();
        a_mat
    }

    /// Converts the matrix into Compressed Sparse Row (CSR) format.
    /// Duplicate entries are summed.
    pub fn to_csr(&self) -> CSR<I, T> {
        let nnz = self.nnz();
        let mut rowptr = vec![I::zero(); self.rows + 1];
        let mut colidx = vec![I::zero(); nnz];
        let mut values = vec![T::zero(); nnz];

        coo_tocsr(
            self.rows,
            self.cols,
            self.nnz(),
            &self.rowidx,
            &self.colidx,
            &self.values,
            &mut rowptr,
            &mut colidx,
            &mut values,
        );

        let mut a_mat = CSR::new(self.rows, self.cols, rowptr, colidx, values).unwrap();
        a_mat.sum_duplicates().unwrap();
        a_mat
    }

    /// Converts the matrix into a dense 2D slice.
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let rows = self.rows;
        let cols = self.cols;
        let mut dense = vec![vec![T::zero(); cols]; rows];
        // for i := range dense {
        // 	dense[i] = make([]float64, mat.cols)
        // }
        for n in 0..self.nnz() {
            let i = self.rowidx[n].to_usize().unwrap();
            let j = self.colidx[n].to_usize().unwrap();
            dense[i][j] = dense[i][j] + self.values[n];
        }
        dense
    }

    /// Provides the upper triangular portion of the matrix in
    /// coordinate format.
    pub fn upper(&self, k: I) -> Coo<I, T> {
        let c = (self.nnz() / 2) + self.cols;
        let mut rowidx = Vec::with_capacity(c);
        let mut colidx = Vec::with_capacity(c);
        let mut values = Vec::with_capacity(c);

        for i in 0..self.nnz() {
            if self.rowidx[i] + k <= self.colidx[i] {
                rowidx.push(self.rowidx[i]);
                colidx.push(self.colidx[i]);
                values.push(self.values[i]);
            }
        }

        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colidx,
            values: values,
        }
    }

    /// Returns a slice of the elements on the main diagonal.
    pub fn diagonal(&self) -> Vec<T> {
        let n = min(self.rows, self.cols);
        let mut d = vec![T::zero(); n];
        for i in 0..self.values.len() {
            let r = self.rowidx[i];
            if r == self.colidx[i] {
                let j = r.to_usize().unwrap();
                d[j] = d[j] + self.values[i];
            }
        }
        d
    }

    pub fn to_string(&self) -> String {
        let mut buf: String = String::new();
        for i in 0..self.nnz() {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&format!(
                "({}, {}) {}",
                self.rowidx[i], self.colidx[i], self.values[i]
            ));
        }
        buf
    }

    /// Creates a new coordinate matrix by vertically stacking the
    /// given input matrices. The matrix arguments must have the same number
    /// of columns.
    pub fn v_stack(a_mat: &Coo<I, T>, b_mat: &Coo<I, T>) -> Result<Coo<I, T>> {
        if a_mat.cols != b_mat.cols {
            return Err(format_err!(
                "cols mismatch {} != {}",
                a_mat.cols,
                b_mat.cols
            ));
        }

        // let mut shifted = vec![I::zero(); b_mat.rowidx.len()];
        // for (i, &r) in b_mat.rowidx.iter().enumerate() {
        //     shifted[i] = r + a_mat.rows;
        // }

        let rows = a_mat.rows + b_mat.rows;
        let cols = a_mat.cols;
        let nnz = a_mat.nnz() + b_mat.nnz();

        let mut rowidx = Vec::with_capacity(nnz);
        rowidx.extend(&a_mat.rowidx);
        // rowidx.extend(&shifted);
        rowidx.extend(
            b_mat
                .rowidx
                .iter()
                .map(|&r| r + I::from(a_mat.rows).unwrap()),
        );

        let mut colidx = Vec::with_capacity(nnz);
        colidx.extend(&a_mat.colidx);
        colidx.extend(&b_mat.colidx);

        let mut values = Vec::with_capacity(nnz);
        values.extend(&a_mat.values);
        values.extend(&b_mat.values);

        Ok(Coo {
            rows,
            cols,
            rowidx,
            colidx,
            values,
        })
    }

    /// Creates a new coordinate matrix by vertically stacking the
    /// given input matrices. The matrix arguments must have the same number
    /// of columns.
    pub fn v_stack3(a_mat: &Coo<I, T>, b_mat: &Coo<I, T>, c_mat: &Coo<I, T>) -> Result<Coo<I, T>> {
        if a_mat.cols != b_mat.cols {
            return Err(format_err!(
                "cols mismatch {} != {}",
                a_mat.cols,
                b_mat.cols
            ));
        }
        if a_mat.cols != c_mat.cols {
            return Err(format_err!(
                "cols mismatch {} != {}",
                a_mat.cols,
                c_mat.cols
            ));
        }

        Coo::v_stack(&Coo::v_stack(a_mat, b_mat)?, c_mat)
    }

    /// Creates a new coordinate matrix by horizontally stacking the
    /// given input matrices. The matrix arguments must have the same number
    /// of rows.
    pub fn h_stack(a_mat: &Coo<I, T>, b_mat: &Coo<I, T>) -> Result<Coo<I, T>> {
        if a_mat.rows != b_mat.rows {
            return Err(format_err!(
                "rows mismatch {} != {}",
                a_mat.rows,
                b_mat.rows
            ));
        }

        // let mut shifted = vec![I::zero(); b_mat.colidx.len()];
        // for (i, &c) in b_mat.colidx.iter().enumerate() {
        //     shifted[i] = shifted[i] + (c + a_mat.cols)
        // }

        let rows = a_mat.rows;
        let cols = a_mat.cols + b_mat.cols;
        let nnz = a_mat.nnz() + b_mat.nnz();

        let mut rowidx = Vec::with_capacity(nnz);
        rowidx.extend(&a_mat.rowidx);
        rowidx.extend(&b_mat.rowidx);

        let mut colidx = Vec::with_capacity(nnz);
        colidx.extend(&a_mat.colidx);
        // colidx.extend(shifted);
        colidx.extend(
            b_mat
                .colidx
                .iter()
                .map(|&c| c + I::from(a_mat.cols).unwrap()),
        );

        let mut values = Vec::with_capacity(nnz);
        values.extend(&a_mat.values);
        values.extend(&b_mat.values);

        Coo::new(rows, cols, rowidx, colidx, values)
    }

    /// Creates a new coordinate matrix by horizontally stacking the
    /// given input matrices. The matrix arguments must have the same number
    /// of rows.
    pub fn h_stack3(a_mat: &Coo<I, T>, b_mat: &Coo<I, T>, c_mat: &Coo<I, T>) -> Result<Coo<I, T>> {
        if a_mat.rows != b_mat.rows {
            return Err(format_err!(
                "rows mismatch {} != {}",
                a_mat.rows,
                b_mat.rows
            ));
        }
        if a_mat.rows != c_mat.rows {
            return Err(format_err!(
                "rows mismatch {} != {}",
                a_mat.rows,
                c_mat.rows
            ));
        }

        Coo::h_stack(&Coo::h_stack(a_mat, b_mat)?, c_mat)
    }

    /// Assembles a matrix from four sub-matrices using HStackCoo and VStackCoo.
    ///
    /// ```text
    ///          -----------
    ///         | J11 | J12 |
    ///     J = |-----------|
    ///         | J21 | J22 |
    ///          -----------
    /// ```
    ///
    /// An error is returned if the matrix dimensions do not match correctly.
    pub fn compose(j: [[&Coo<I, T>; 2]; 2]) -> Result<Coo<I, T>> {
        let j1x = Coo::h_stack(j[0][0], j[0][1])?;
        let j2x = Coo::h_stack(j[1][0], j[1][1])?;
        Coo::v_stack(&j1x, &j2x)
    }

    /// Assembles a matrix from four sub-matrices using HStackCoo and VStackCoo.
    ///
    /// ```text
    ///          -----------------
    ///         | J11 | J12 | J13 |
    ///         |-----------|-----|
    ///     J = | J21 | J22 | J23 |
    ///         |-----------------|
    ///         | J31 | J32 | J33 |
    ///          -----------------
    /// ```
    ///
    /// An error is returned if the matrix dimensions do not match correctly.
    pub fn compose3(j: [[&Coo<I, T>; 3]; 3]) -> Result<Coo<I, T>> {
        let j1x = Coo::h_stack3(j[0][0], j[0][1], j[0][2])?;
        let j2x = Coo::h_stack3(j[1][0], j[1][1], j[1][2])?;
        let j3x = Coo::h_stack3(j[2][0], j[2][1], j[2][2])?;
        Coo::v_stack3(&j1x, &j2x, &j3x)
    }
}
