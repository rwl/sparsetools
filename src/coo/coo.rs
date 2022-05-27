use crate::coord::coo_tocsr;
use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use std::cmp::min;

/// A sparse matrix with scalar values stored in Coordinate format
/// (also called "aij", "ijv" or "triplet" format).
pub struct Coo<I: Integer, T: Scalar> {
    pub rows: I,
    pub cols: I,
    /// Row indexes (size nnz).
    pub rowidx: Vec<I>,
    /// Column indexes (size nnz).
    pub colidx: Vec<I>,
    /// Explicitly stored values (size nnz).
    pub data: Vec<T>,
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
            rowidx: vec![I::zero(); nnz.to_usize().unwrap()],
            colidx: vec![I::zero(); nnz.to_usize().unwrap()],
            data: vec![T::zero(); nnz.to_usize().unwrap()],
        }
    }

    /// Creates a sparse matrix in coordinate format from a full matrix.
    pub fn from_dense(a: &[&[T]]) -> Self {
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

    pub fn set(&mut self, row: I, col: I, v: T) {
        self.rowidx.push(row);
        self.colidx.push(col);
        self.data.push(v);
    }

    pub fn append(&mut self, row: &[I], col: &[I], v: &[T]) {
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
    fn nnz(&self) -> I {
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
    pub fn transpose(&self) -> Coo<I, T> {
        Coo {
            rows: self.cols,
            cols: self.rows,
            rowidx: self.colidx.clone(),
            colidx: self.rowidx.clone(),
            data: self.data.clone(),
        }
    }

    /// An alias for `transpose`.
    pub fn t(&self) -> Coo<I, T> {
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

        let mut A = CSC {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colptr,
            data,
        };
        A.sum_duplicates();
        A
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

        let mut A = CSR {
            rows: self.rows,
            cols: self.cols,
            rowptr,
            colidx,
            data,
        };
        A.sum_duplicates();
        A
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

    /// Creates a CSR matrix that is the element-wise sum of the
    /// receiver and the given matrix.
    pub fn add(&self, m: &Coo<I, T>) -> CSR<I, T> {
        // if m == nil {
        // 	return mat.ToCSR()
        // }
        let k = self.nnz();
        let nnz = (k + m.nnz()).to_usize().unwrap();

        // let rowidx = vec![I::zero(); nnz];
        let mut rowidx = Vec::with_capacity(nnz);
        // let colidx = vec![I::zero(); nnz];
        let mut colidx = Vec::with_capacity(nnz);
        // let data = vec![T::zero(); nnz];
        let mut data = Vec::with_capacity(nnz);

        // copy(rowidx, self.rowidx);
        rowidx.extend(&self.rowidx);
        // copy(colidx, self.colidx);
        colidx.extend(&self.colidx);
        // copy(data, self.data);
        data.extend(&self.data);

        // copy(rowidx[k..], m.rowidx);
        rowidx.extend(&m.rowidx);
        // copy(colidx[k..], m.colidx);
        colidx.extend(&m.colidx);
        // copy(data[k..], m.data);
        data.extend(&m.data);

        let A = Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colidx,
            data,
        };
        A.to_csr() // Duplicate entries are summed.
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
    pub fn v_stack(A: &Coo<I, T>, B: &Coo<I, T>) -> Result<Coo<I, T>, String> {
        if A.cols != B.cols {
            return Err(format!("cols mismatch {} != {}", A.cols, B.cols));
        }

        let mut shifted = vec![I::zero(); B.rowidx.len()];
        for (i, &r) in B.rowidx.iter().enumerate() {
            shifted[i] = r + A.rows;
        }

        let rows = A.rows + B.rows;
        let cols = A.cols;
        let nnz = (A.nnz() + B.nnz()).to_usize().unwrap();

        let mut rowidx = Vec::with_capacity(nnz);
        rowidx.extend(&A.rowidx);
        rowidx.extend(&shifted);

        let mut colidx = Vec::with_capacity(nnz);
        colidx.extend(&A.colidx);
        colidx.extend(&B.colidx);

        let mut data = Vec::with_capacity(nnz);
        data.extend(&A.data);
        data.extend(&B.data);

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
    pub fn h_stack(A: &Coo<I, T>, B: &Coo<I, T>) -> Result<Coo<I, T>, String> {
        if A.rows != B.rows {
            return Err(format!("rows mismatch {} != {}", A.rows, B.rows));
        }

        let mut shifted = vec![I::zero(); B.colidx.len()];
        for (i, &c) in B.colidx.iter().enumerate() {
            shifted[i] = shifted[i] + (c + A.cols)
        }

        let rows = A.rows;
        let cols = A.cols + B.cols;
        let nnz = (A.nnz() + B.nnz()).to_usize().unwrap();

        let mut rowidx = Vec::with_capacity(nnz);
        rowidx.extend(&A.rowidx);
        rowidx.extend(&B.rowidx);

        let mut colidx = Vec::with_capacity(nnz);
        colidx.extend(&A.colidx);
        colidx.extend(shifted);

        let mut data = Vec::with_capacity(nnz);
        data.extend(&A.data);
        data.extend(&B.data);

        Coo::new(rows, cols, rowidx, colidx, data)
    }

    /// Assembles a matrix from four sub-matrices using HStackCoo and VStackCoo.
    ///        -----------
    ///       | J11 | J12 |
    ///   J = |-----------|
    ///       | J21 | J22 |
    ///        -----------
    /// An error is returned if the matrix dimensions do not match correctly.
    pub fn compose(
        J11: &Coo<I, T>,
        J12: &Coo<I, T>,
        J21: &Coo<I, T>,
        J22: &Coo<I, T>,
    ) -> Result<Coo<I, T>, String> {
        let J1X = Coo::h_stack(J11, J12)?;
        let J2X = Coo::h_stack(J21, J22)?;
        Coo::v_stack(&J1X, &J2X)
    }
}
