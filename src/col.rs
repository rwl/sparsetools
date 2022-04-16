use crate::dense::axpy;
use crate::scalar::Scalar;
use crate::{csr_diagonal, csr_matmat, csr_matmat_maxnnz, csr_tocsc};
use num_traits::{PrimInt, Signed};
use std::ops::AddAssign;

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
pub fn csc_matvec<I: PrimInt, T: Scalar + AddAssign>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Ai: &[I],
    Ax: &[T],
    Xx: &[T],
    Yx: &mut [T],
) {
    for j in 0..n_col.to_usize().unwrap() {
        let col_start = Ap[j].to_usize().unwrap();
        let col_end = Ap[j + 1].to_usize().unwrap();

        for ii in col_start..col_end {
            let i = Ai[ii].to_usize().unwrap();
            Yx[i] += Ax[ii] * Xx[j];
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
pub fn csc_matvecs<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    n_vecs: I,
    Ap: &[I],
    Ai: &[I],
    Ax: &[T],
    Xx: &[T],
    Yx: &mut [T],
) {
    for j in 0..n_col.to_usize().unwrap() {
        let start = Ap[j].to_usize().unwrap();
        let end = Ap[j + 1].to_usize().unwrap();
        for ii in start..end {
            let i = Ai[ii].to_usize().unwrap();
            axpy(
                n_vecs,
                Ax[ii],
                // Xx + /*(npy_intp)*/n_vecs * j,
                &Xx[(n_vecs.to_usize().unwrap() * j)..],
                // Yx + /*(npy_intp)*/n_vecs * i,
                &mut Yx[(n_vecs.to_usize().unwrap() * i)..],
            );
        }
    }
}

// Derived methods //

pub fn csc_diagonal<I: PrimInt + Signed, T: Scalar>(
    k: isize,
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Yx: &mut [T],
) {
    csr_diagonal(-k, n_col, n_row, Ap, Aj, Ax, Yx);
}

pub fn csc_tocsr<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Ai: &[I],
    Ax: &[T],
    Bp: &mut [I],
    Bj: &mut [I],
    Bx: &mut [T],
) {
    csr_tocsc(n_col, n_row, Ap, Ai, Ax, Bp, Bj, Bx);
}

pub fn csc_matmat_maxnnz<I: PrimInt>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Ai: &[I],
    Bp: &[I],
    Bi: &[I],
) -> usize /*npy_intp*/
{
    return csr_matmat_maxnnz(n_col, n_row, Bp, Bi, Ap, Ai);
}

pub fn csc_matmat<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Ai: &[I],
    Ax: &[T],
    Bp: &[I],
    Bi: &[I],
    Bx: &[T],
    Cp: &mut [I],
    Ci: &mut [I],
    Cx: &mut [T],
) {
    csr_matmat(n_col, n_row, Bp, Bi, Bx, Ap, Ai, Ax, Cp, Ci, Cx);
}
