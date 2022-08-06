use crate::coord::coo_tocsr;
use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use std::cmp::min;
use std::ops::Neg;

/// A sparse matrix with scalar values stored in Coordinate format
/// (also called "aij", "ijv" or "triplet" format).
#[derive(Clone)]
pub struct Coo<I, T> {
    pub(crate) rows: I,
    pub(crate) cols: I,
    pub(crate) rowidx: Vec<I>,
    pub(crate) colidx: Vec<I>,
    pub(crate) data: Vec<T>,
}

impl<I: Integer, T: Scalar> Coo<I, T> {
    /// Creates a new coordinate matrix. Inputs are not copied. An error
    /// is returned if the slice arguments do not have the same length.
    pub fn new(
        rows: I,
        cols: I,
        rowidx: Vec<I>,
        colidx: Vec<I>,
        data: Vec<T>,
    ) -> Result<Self, String> {
        let nnz = data.len();
        if nnz != rowidx.len() || nnz != colidx.len() {
            return Err("row, column, and data array must all be the same length".to_string());
        }
        Ok(Self {
            rows,
            cols,
            rowidx,
            colidx,
            data,
        })
    }

    pub fn empty(rows: I, cols: I, nnz: I) -> Self {
        Self {
            rows,
            cols,
            rowidx: Vec::with_capacity(nnz.to_usize().unwrap()),
            colidx: Vec::with_capacity(nnz.to_usize().unwrap()),
            data: Vec::with_capacity(nnz.to_usize().unwrap()),
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
        let mut data = Vec::<T>::with_capacity(m * n);

        for (i, r) in a.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                if v != T::zero() {
                    rowidx.push(I::from(i).unwrap());
                    colidx.push(I::from(j).unwrap());
                    data.push(v);
                }
            }
        }

        Self {
            rows: I::from(m).unwrap(),
            cols: I::from(n).unwrap(),
            rowidx,
            colidx,
            data,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> I {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> I {
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
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn push(&mut self, row: I, col: I, v: T) {
        self.rowidx.push(row);
        self.colidx.push(col);
        self.data.push(v);
    }

    pub fn extend(&mut self, row: &[I], col: &[I], v: &[T]) {
        self.rowidx.extend(row);
        self.colidx.extend(col);
        self.data.extend(v);
    }

    /// Diagonal creates a coordinate matrix with the given diagonal.
    pub fn with_diagonal(diag: &[T]) -> Self {
        let n = diag.len();
        let mut rowidx = vec![I::zero(); n];
        let mut colidx = vec![I::zero(); n];
        let mut data = vec![T::zero(); n];
        for i in 0..n {
            rowidx[i] = I::from(i).unwrap();
            colidx[i] = I::from(i).unwrap();
            data[i] = diag[i];
        }
        Self {
            rows: I::from(n).unwrap(),
            cols: I::from(n).unwrap(),
            rowidx,
            colidx,
            data,
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
    //     let mut data = vec![T::zero(); nnz];
    //     for i in 0..data.len() {
    //         data[i] = rng.gen();
    //     }
    //
    //     Self {
    //         rows,
    //         cols,
    //         rowidx,
    //         colidx,
    //         data,
    //     }
    // }

    pub fn identity(n: I) -> Self {
        let mut rowidx = vec![I::zero(); n.to_usize().unwrap()];
        let mut colidx = vec![I::zero(); n.to_usize().unwrap()];
        let mut data = vec![T::zero(); n.to_usize().unwrap()];
        for i in 0..n.to_usize().unwrap() {
            rowidx[i] = I::from(i).unwrap();
            colidx[i] = I::from(i).unwrap();
            data[i] = T::one();
        }
        Self {
            rows: n,
            cols: n,
            rowidx,
            colidx,
            data,
        }
    }

    /// Returns the count of explicitly stored values (nonzeros).
    pub fn nnz(&self) -> I {
        I::from(self.data.len()).unwrap()
    }

    // Copy creates an identical coordinate matrix with newly allocated
    // index and data slices.
    // func (mat *Coo) Copy() *Coo {
    // 	nnz := mat.NNZ()
    //
    // 	rowidx := make([]int, nnz)
    // 	colidx := make([]int, nnz)
    // 	data := make([]float64, nnz)
    // 	copy(rowidx, mat.rowidx)
    // 	copy(colidx, mat.colidx)
    // 	copy(data, mat.data)
    //
    // 	return &Coo{rows: mat.rows, cols: mat.cols,
    // 		rowidx: rowidx, colidx: colidx, data: data}
    // }

    /// Creates a coordinate matrix that is the transpose of the
    /// receiver. The underlying index and data slices are not copied.
    pub fn transpose(self) -> Coo<I, T> {
        Coo {
            rows: self.cols,
            cols: self.rows,
            rowidx: self.colidx,
            colidx: self.rowidx,
            data: self.data,
        }
    }

    /// An alias for `transpose`.
    pub fn t(self) -> Coo<I, T> {
        self.transpose()
    }

    /// Converts the matrix to Compressed Sparse Column (CSC) format.
    /// Duplicate entries are summed.
    pub fn to_csc(&self) -> CSC<I, T> {
        let nnz = self.nnz().to_usize().unwrap();
        let mut rowidx = vec![I::zero(); nnz];
        let mut colptr = vec![I::zero(); self.cols.to_usize().unwrap() + 1];
        let mut data = vec![T::zero(); nnz];

        coo_tocsr(
            self.cols,
            self.rows,
            self.nnz(),
            &self.colidx,
            &self.rowidx,
            &self.data,
            &mut colptr,
            &mut rowidx,
            &mut data,
        );

        let mut a_mat = CSC {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colptr,
            data,
        };
        a_mat.sum_duplicates();
        a_mat
    }

    /// Converts the matrix into Compressed Sparse Row (CSR) format.
    /// Duplicate entries are summed.
    pub fn to_csr(&self) -> CSR<I, T> {
        let nnz = self.nnz().to_usize().unwrap();
        let mut rowptr = vec![I::zero(); self.rows.to_usize().unwrap() + 1];
        let mut colidx = vec![I::zero(); nnz];
        let mut data = vec![T::zero(); nnz];

        coo_tocsr(
            self.rows,
            self.cols,
            self.nnz(),
            &self.rowidx,
            &self.colidx,
            &self.data,
            &mut rowptr,
            &mut colidx,
            &mut data,
        );

        let mut a_mat = CSR {
            rows: self.rows,
            cols: self.cols,
            rowptr,
            colidx,
            data,
        };
        a_mat.sum_duplicates().unwrap();
        a_mat
    }

    /// Converts the matrix into a dense 2D slice.
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let rows = self.rows.to_usize().unwrap();
        let cols = self.cols.to_usize().unwrap();
        let mut dense = vec![vec![T::zero(); cols]; rows];
        // for i := range dense {
        // 	dense[i] = make([]float64, mat.cols)
        // }
        for n in 0..self.nnz().to_usize().unwrap() {
            let i = self.rowidx[n].to_usize().unwrap();
            let j = self.colidx[n].to_usize().unwrap();
            dense[i][j] = dense[i][j] + self.data[n];
        }
        dense
    }

    /// Provides the upper triangular portion of the matrix in
    /// coordinate format.
    pub fn upper(&self, k: I) -> Coo<I, T> {
        let c = (self.nnz().to_usize().unwrap() / 2) + self.cols.to_usize().unwrap();
        let mut rowidx = Vec::with_capacity(c);
        let mut colidx = Vec::with_capacity(c);
        let mut data = Vec::with_capacity(c);

        for i in 0..self.nnz().to_usize().unwrap() {
            if self.rowidx[i] + k <= self.colidx[i] {
                rowidx.push(self.rowidx[i]);
                colidx.push(self.colidx[i]);
                data.push(self.data[i]);
            }
        }

        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colidx,
            data,
        }
    }

    /// Returns a slice of the elements on the main diagonal.
    pub fn diagonal(&self) -> Vec<T> {
        let n = min(self.rows, self.cols).to_usize().unwrap();
        let mut d = vec![T::zero(); n];
        for i in 0..self.data.len() {
            let r = self.rowidx[i];
            if r == self.colidx[i] {
                let j = r.to_usize().unwrap();
                d[j] = d[j] + self.data[i];
            }
        }
        d
    }

    pub fn to_string(&self) -> String {
        let mut buf: String = String::new();
        for i in 0..self.nnz().to_usize().unwrap() {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&format!(
                "({}, {}) {}",
                self.rowidx[i], self.colidx[i], self.data[i]
            ));
        }
        buf
    }

    /// Creates a new coordinate matrix by vertically stacking the
    /// given input matrices. The matrix arguments must have the same number
    /// of columns.
    pub fn v_stack(
        a_mat: &Coo<I, T>,
        b_mat: &Coo<I, T>,
        c_mat: Option<&Coo<I, T>>,
    ) -> Result<Coo<I, T>, String> {
        if a_mat.cols != b_mat.cols {
            return Err(format!("cols mismatch {} != {}", a_mat.cols, b_mat.cols));
        }
        if let Some(c_mat) = c_mat {
            if a_mat.cols != c_mat.cols {
                return Err(format!("cols mismatch {} != {}", a_mat.cols, c_mat.cols));
            }
        }

        // let mut shifted = vec![I::zero(); b_mat.rowidx.len()];
        // for (i, &r) in b_mat.rowidx.iter().enumerate() {
        //     shifted[i] = r + a_mat.rows;
        // }

        let mut rows = a_mat.rows + b_mat.rows;
        let cols = a_mat.cols;
        let mut nnz = (a_mat.nnz() + b_mat.nnz()).to_usize().unwrap();
        if let Some(c_mat) = c_mat {
            rows += c_mat.rows;
            nnz += c_mat.nnz().to_usize().unwrap();
        }

        let mut rowidx = Vec::with_capacity(nnz);
        rowidx.extend(&a_mat.rowidx);
        // rowidx.extend(&shifted);
        rowidx.extend(b_mat.rowidx.iter().map(|&r| r + a_mat.rows));

        let mut colidx = Vec::with_capacity(nnz);
        colidx.extend(&a_mat.colidx);
        colidx.extend(&b_mat.colidx);

        let mut data = Vec::with_capacity(nnz);
        data.extend(&a_mat.data);
        data.extend(&b_mat.data);

        if let Some(c_mat) = c_mat {
            rowidx.extend(c_mat.rowidx.iter().map(|&r| r + a_mat.rows + b_mat.rows));
            colidx.extend(&c_mat.colidx);
            data.extend(&c_mat.data);
        }

        Ok(Coo {
            rows,
            cols,
            rowidx,
            colidx,
            data,
        })
    }

    /// Creates a new coordinate matrix by horizontally stacking the
    /// given input matrices. The matrix arguments must have the same number
    /// of rows.
    pub fn h_stack(
        a_mat: &Coo<I, T>,
        b_mat: &Coo<I, T>,
        c_mat: Option<&Coo<I, T>>,
    ) -> Result<Coo<I, T>, String> {
        if a_mat.rows != b_mat.rows {
            return Err(format!("rows mismatch {} != {}", a_mat.rows, b_mat.rows));
        }
        if let Some(c_mat) = c_mat {
            if a_mat.rows != c_mat.rows {
                return Err(format!("rows mismatch {} != {}", a_mat.rows, c_mat.rows));
            }
        }

        // let mut shifted = vec![I::zero(); b_mat.colidx.len()];
        // for (i, &c) in b_mat.colidx.iter().enumerate() {
        //     shifted[i] = shifted[i] + (c + a_mat.cols)
        // }

        let rows = a_mat.rows;
        let mut cols = a_mat.cols + b_mat.cols;
        let mut nnz = (a_mat.nnz() + b_mat.nnz()).to_usize().unwrap();
        if let Some(c_mat) = c_mat {
            cols += c_mat.cols;
            nnz += c_mat.nnz().to_usize().unwrap();
        }

        let mut rowidx = Vec::with_capacity(nnz);
        rowidx.extend(&a_mat.rowidx);
        rowidx.extend(&b_mat.rowidx);

        let mut colidx = Vec::with_capacity(nnz);
        colidx.extend(&a_mat.colidx);
        // colidx.extend(shifted);
        colidx.extend(b_mat.colidx.iter().map(|&c| c + a_mat.cols));

        let mut data = Vec::with_capacity(nnz);
        data.extend(&a_mat.data);
        data.extend(&b_mat.data);

        if let Some(c_mat) = c_mat {
            rowidx.extend(&c_mat.rowidx);
            colidx.extend(c_mat.colidx.iter().map(|&c| c + b_mat.cols + a_mat.cols));
            data.extend(&c_mat.data);
        }

        Coo::new(rows, cols, rowidx, colidx, data)
    }

    /// Assembles a matrix from four sub-matrices using HStackCoo and VStackCoo.
    ///
    ///          -----------
    ///         | J11 | J12 |
    ///     J = |-----------|
    ///         | J21 | J22 |
    ///          -----------
    ///
    /// An error is returned if the matrix dimensions do not match correctly.
    pub fn compose(j: [[&Coo<I, T>; 2]; 2]) -> Result<Coo<I, T>, String> {
        let j1x = Coo::h_stack(j[0][0], j[0][1], None)?;
        let j2x = Coo::h_stack(j[1][0], j[1][1], None)?;
        Coo::v_stack(&j1x, &j2x, None)
    }

    /// Assembles a matrix from four sub-matrices using HStackCoo and VStackCoo.
    ///
    ///          -----------------
    ///         | J11 | J12 | J13 |
    ///         |-----------|-----|
    ///     J = | J21 | J22 | J23 |
    ///         |-----------------|
    ///         | J31 | J32 | J33 |
    ///          -----------------
    ///
    /// An error is returned if the matrix dimensions do not match correctly.
    pub fn compose3(j: [[&Coo<I, T>; 3]; 3]) -> Result<Coo<I, T>, String> {
        let j1x = Coo::h_stack(j[0][0], j[0][1], Some(j[0][2]))?;
        let j2x = Coo::h_stack(j[1][0], j[1][1], Some(j[1][2]))?;
        let j3x = Coo::h_stack(j[2][0], j[2][1], Some(j[2][2]))?;
        Coo::v_stack(&j1x, &j2x, Some(&j3x))
    }
}

impl<I: Integer, T: Scalar + Neg<Output = T>> Neg for Coo<I, T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx: self.rowidx.clone(),
            colidx: self.colidx.clone(),
            data: self.data.iter().map(|&d| -d).collect(),
        }
    }
}
