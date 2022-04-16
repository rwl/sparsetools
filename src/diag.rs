use crate::Scalar;
use num_traits::PrimInt;

/// Compute Y += A*X for DIA matrix A and dense vectors X,Y
///
///
/// Input Arguments:
///   I  n_row            - number of rows in A
///   I  n_col            - number of columns in A
///   I  n_diags          - number of diagonals
///   I  L                - length of each diagonal
///   I  offsets[n_diags] - diagonal offsets
///   T  diags[n_diags,L] - nonzeros
///   T  Xx[n_col]        - input vector
///
/// Output Arguments:
///   T  Yx[n_row]        - output vector
///
/// Note:
///   Output array Yx must be preallocated
///   Negative offsets correspond to lower diagonals
///   Positive offsets correspond to upper diagonals
pub fn dia_matvec<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    n_diags: I,
    L: I,
    offsets: &[I],
    diags: &[T],
    Xx: &[T],
    Yx: &mut [T],
) {
    for i in 0..n_diags {
        let k: I = offsets[i]; //diagonal offset

        let i_start: I = I::max(0, -k);
        let j_start: I = I::max(0, k);
        let j_end: I = I::min(I::min(n_row + k, n_col), L);

        let N: I = j_end - j_start; //number of elements to process

        let diag = diags + /*(npy_intp)*/i*L + j_start;
        let x = Xx + j_start;
        let y = Yx + i_start;

        for n in 0..N {
            y[n] += diag[n] * x[n];
        }
    }
}
