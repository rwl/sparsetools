use crate::Scalar;
use num_traits::PrimInt;

/// Compute B = A for COO matrix A, CSR matrix B
///
///
/// Input Arguments:
///   I  n_row      - number of rows in A
///   I  n_col      - number of columns in A
///   I  nnz        - number of nonzeros in A
///   I  Ai[nnz(A)] - row indices
///   I  Aj[nnz(A)] - column indices
///   T  Ax[nnz(A)] - nonzeros
/// Output Arguments:
///   I Bp  - row pointer
///   I Bj  - column indices
///   T Bx  - nonzeros
///
/// Note:
///   Output arrays Bp, Bj, and Bx must be preallocated
///
/// Note:
///   Input:  row and column indices *are not* assumed to be ordered
///
///   Note: duplicate entries are carried over to the CSR represention
///
///   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
///
pub fn coo_tocsr<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    nnz: I,
    Ai: &[I],
    Aj: &[I],
    Ax: &[T],
    Bp: &mut [I],
    Bj: &mut [I],
    Bx: &mut [T],
) {
    //compute number of non-zero entries per row of A
    // fill(Bp, Bp + n_row, 0);
    Bp.fill(I::zero());

    for n in 0..nnz {
        Bp[Ai[n]] += 1;
    }

    //cumsum the nnz per row to get Bp[]
    let mut cumsum = 0;
    for i in 0..n_row {
        let temp: I = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz;

    //write Aj,Ax into Bj,Bx
    for n in 0..nnz {
        let row: I = Ai[n];
        let dest: I = Bp[row];

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row] += 1;
    }

    let mut last = 0;
    for i in 0..=n_row {
        let temp: I = Bp[i];
        Bp[i] = last;
        last = temp;
    }

    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

/// Compute B += A for COO matrix A, dense matrix B
///
/// Input Arguments:
///   I  n_row           - number of rows in A
///   I  n_col           - number of columns in A
///   npy_int64  nnz     - number of nonzeros in A
///   I  Ai[nnz(A)]      - row indices
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - nonzeros
///   T  Bx[n_row*n_col] - dense matrix
pub fn coo_todense<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    nnz: usize, /*npy_int64*/
    Ai: &[I],
    Aj: &[I],
    Ax: &[T],
    Bx: &mut [T],
    fortran: bool,
) {
    if !fortran {
        for n in 0..nnz {
            Bx[ /*(npy_intp)*/n_col * Ai[n] + Aj[n] ] += Ax[n];
        }
    } else {
        for n in 0..nnz {
            Bx[ /*(npy_intp)*/n_row * Aj[n] + Ai[n] ] += Ax[n];
        }
    }
}

/// Compute Y += A*X for COO matrix A and dense vectors X,Y
///
///
/// Input Arguments:
///   npy_int64  nnz   - number of nonzeros in A
///   I  Ai[nnz]       - row indices
///   I  Aj[nnz]       - column indices
///   T  Ax[nnz]       - nonzero values
///   T  Xx[n_col]     - input vector
///
/// Output Arguments:
///   T  Yx[n_row]     - output vector
///
/// Notes:
///   Output array Yx must be preallocated
///
///   Complexity: Linear.  Specifically O(nnz(A))
pub fn coo_matvec<I: PrimInt, T: Scalar>(
    nnz: usize, /*npy_int64*/
    Ai: &[I],
    Aj: &[I],
    Ax: &[T],
    Xx: &[T],
    Yx: &mut [T],
) {
    for n in 0..nnz {
        Yx[Ai[n]] += Ax[n] * Xx[Aj[n]];
    }
}
