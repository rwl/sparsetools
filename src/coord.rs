use crate::traits::{Integer, Scalar};

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
pub fn coo_tocsr<I: Integer, T: Scalar>(
    n_row: I,
    _n_col: I,
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

    for n in 0..nnz.to_usize().unwrap() {
        let i = Ai[n].to_usize().unwrap();
        Bp[i] += I::one();
    }

    //cumsum the nnz per row to get Bp[]
    let mut cumsum = I::zero();
    for i in 0..n_row.to_usize().unwrap() {
        let temp: I = Bp[i];
        Bp[i] = cumsum;
        cumsum = cumsum + temp;
    }
    Bp[n_row.to_usize().unwrap()] = nnz;

    //write Aj,Ax into Bj,Bx
    for n in 0..nnz.to_usize().unwrap() {
        let row = Ai[n].to_usize().unwrap();
        let dest = Bp[row].to_usize().unwrap();

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row] += I::one();
    }

    let mut last = I::zero();
    for i in 0..=n_row.to_usize().unwrap() {
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
pub fn coo_todense<I: Integer, T: Scalar>(
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
            let i = /*(npy_intp)*/(n_col * Ai[n] + Aj[n]).to_usize().unwrap();
            Bx[i] += Ax[n];
        }
    } else {
        for n in 0..nnz {
            let i = /*(npy_intp)*/(n_row * Aj[n] + Ai[n]).to_usize().unwrap();
            Bx[i] += Ax[n];
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
pub fn coo_matvec<I: Integer, T: Scalar>(
    nnz: usize, /*npy_int64*/
    Ai: &[I],
    Aj: &[I],
    Ax: &[T],
    Xx: &[T],
    Yx: &mut [T],
) {
    for n in 0..nnz {
        let i = Ai[n].to_usize().unwrap();
        let j = Aj[n].to_usize().unwrap();
        Yx[i] += Ax[n] * Xx[j];
    }
}
