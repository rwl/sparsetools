use crate::dense::axpy;
use crate::traits::{Integer, Scalar};
use crate::{csr_diagonal, csr_matmat, csr_matmat_maxnnz, csr_tocsc};

/// Compute Y += A*X for CSC matrix A and dense vectors X,Y
///
///
/// Input Arguments:
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - column pointer
///   I  Ai[nnz(A)]    - row indices
///   T  Ax[n_col]     - nonzeros
///   T  Xx[n_col]     - input vector
///
/// Output Arguments:
///   T  Yx[n_row]     - output vector
///
/// Note:
///   Output array Yx must be preallocated
///
///   Complexity: Linear.  Specifically O(nnz(A) + n_col)
pub fn csc_matvec<I: Integer, T: Scalar>(
    _n_row: I,
    n_col: I,
    a_p: &[I],
    a_i: &[I],
    a_x: &[T],
    x_x: &[T],
    y_x: &mut [T],
) {
    for j in 0..n_col.to_usize().unwrap() {
        let col_start = a_p[j].to_usize().unwrap();
        let col_end = a_p[j + 1].to_usize().unwrap();

        for ii in col_start..col_end {
            let i = a_i[ii].to_usize().unwrap();
            y_x[i] += a_x[ii] * x_x[j];
        }
    }
}

/// Compute Y += A*X for CSC matrix A and dense block vectors X,Y
///
///
/// Input Arguments:
///   I  n_row            - number of rows in A
///   I  n_col            - number of columns in A
///   I  n_vecs           - number of column vectors in X and Y
///   I  Ap[n_row+1]      - row pointer
///   I  Aj[nnz(A)]       - column indices
///   T  Ax[nnz(A)]       - nonzeros
///   T  Xx[n_col,n_vecs] - input vector
///
/// Output Arguments:
///   T  Yx[n_row,n_vecs] - output vector
///
/// Note:
///   Output array Yx must be preallocated
pub fn csc_matvecs<I: Integer, T: Scalar>(
    _n_row: I,
    n_col: I,
    n_vecs: I,
    a_p: &[I],
    a_i: &[I],
    a_x: &[T],
    x_x: &[T],
    y_x: &mut [T],
) {
    for j in 0..n_col.to_usize().unwrap() {
        let start = a_p[j].to_usize().unwrap();
        let end = a_p[j + 1].to_usize().unwrap();
        for ii in start..end {
            let i = a_i[ii].to_usize().unwrap();
            axpy(
                n_vecs,
                a_x[ii],
                // Xx + /*(npy_intp)*/n_vecs * j,
                &x_x[(n_vecs.to_usize().unwrap() * j)..],
                // Yx + /*(npy_intp)*/n_vecs * i,
                &mut y_x[(n_vecs.to_usize().unwrap() * i)..],
            );
        }
    }
}

// Derived methods //

pub fn csc_diagonal<I: Integer, T: Scalar>(
    k: isize,
    n_row: I,
    n_col: I,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    y_x: &mut [T],
) {
    csr_diagonal(-k, n_col, n_row, a_p, a_j, a_x, y_x);
}

pub fn csc_tocsr<I: Integer, T: Scalar>(
    n_row: I,
    n_col: I,
    a_p: &[I],
    a_i: &[I],
    a_x: &[T],
    b_p: &mut [I],
    b_j: &mut [I],
    b_x: &mut [T],
) {
    csr_tocsc(n_col, n_row, a_p, a_i, a_x, b_p, b_j, b_x);
}

pub fn csc_matmat_maxnnz<I: Integer>(
    n_row: I,
    n_col: I,
    a_p: &[I],
    a_i: &[I],
    b_p: &[I],
    b_i: &[I],
) -> usize /*npy_intp*/
{
    return csr_matmat_maxnnz(n_col, n_row, b_p, b_i, a_p, a_i);
}

pub fn csc_matmat<I: Integer, T: Scalar>(
    n_row: I,
    n_col: I,
    a_p: &[I],
    a_i: &[I],
    a_x: &[T],
    b_p: &[I],
    b_i: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_i: &mut [I],
    c_x: &mut [T],
) {
    csr_matmat(n_col, n_row, b_p, b_i, b_x, a_p, a_i, a_x, c_p, c_i, c_x);
}
