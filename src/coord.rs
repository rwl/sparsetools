use crate::traits::{Integer, Scalar};

/// Compute B = A for COO matrix A, CSR matrix B
///
/// # Input Arguments
/// ```txt
///   I  n_row      - number of rows in A
///   I  n_col      - number of columns in A
///   I  nnz        - number of nonzeros in A
///   I  Ai[nnz(A)] - row indices
///   I  Aj[nnz(A)] - column indices
///   T  Ax[nnz(A)] - nonzeros
/// ```
/// # Output Arguments
/// ```txt
///   I Bp  - row pointer
///   I Bj  - column indices
///   T Bx  - nonzeros
/// ```
///
/// # Notes
///
/// Output arrays Bp, Bj, and Bx must be preallocated.
///
/// Input: row and column indices *are not* assumed to be ordered.
///
/// Duplicate entries are carried over to the CSR represention.
///
/// Complexity: Linear. Specifically `O(nnz(A) + max(n_row,n_col))`.
pub fn coo_tocsr<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    nnz: usize,
    a_i: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &mut [I],
    b_j: &mut [I],
    b_x: &mut [T],
) {
    //compute number of non-zero entries per row of A
    // fill(Bp, Bp + n_row, 0);
    b_p.fill(I::zero());

    for n in 0..nnz {
        let i = a_i[n].to_usize().unwrap();
        b_p[i] += I::one();
    }

    //cumsum the nnz per row to get Bp[]
    let mut cumsum = I::zero();
    for i in 0..n_row {
        let temp: I = b_p[i];
        b_p[i] = cumsum;
        cumsum = cumsum + temp;
    }
    b_p[n_row] = I::from(nnz).unwrap();

    //write Aj,Ax into Bj,Bx
    for n in 0..nnz {
        let row = a_i[n].to_usize().unwrap();
        let dest = b_p[row].to_usize().unwrap();

        b_j[dest] = a_j[n];
        b_x[dest] = a_x[n];

        b_p[row] += I::one();
    }

    let mut last = I::zero();
    for i in 0..=n_row {
        let temp: I = b_p[i];
        b_p[i] = last;
        last = temp;
    }

    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

/// Compute B += A for COO matrix A, dense matrix B
///
/// # Input Arguments
/// ```txt
///   I  n_row           - number of rows in A
///   I  n_col           - number of columns in A
///   npy_int64  nnz     - number of nonzeros in A
///   I  Ai[nnz(A)]      - row indices
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - nonzeros
///   T  Bx[n_row*n_col] - dense matrix
/// ```
pub fn coo_todense<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    nnz: usize, /*npy_int64*/
    a_i: &[I],
    a_j: &[I],
    a_x: &[T],
    b_x: &mut [T],
    fortran: bool,
) {
    if !fortran {
        for n in 0..nnz {
            let i = /*(npy_intp)*/ n_col * a_i[n].to_usize().unwrap() + a_j[n].to_usize().unwrap();
            b_x[i] += a_x[n];
        }
    } else {
        for n in 0..nnz {
            let i = /*(npy_intp)*/ n_row * a_j[n].to_usize().unwrap() + a_i[n].to_usize().unwrap();
            b_x[i] += a_x[n];
        }
    }
}

/// Compute Y += A*X for COO matrix A and dense vectors X,Y
///
/// # Input Arguments
/// ```txt
///   npy_int64  nnz   - number of nonzeros in A
///   I  Ai[nnz]       - row indices
///   I  Aj[nnz]       - column indices
///   T  Ax[nnz]       - nonzero values
///   T  Xx[n_col]     - input vector
/// ```
/// # Output Arguments
/// ```txt
///   T  Yx[n_row]     - output vector
/// ```
///
/// # Notes
///
/// Output array Yx must be preallocated
///
/// Complexity: Linear. Specifically `O(nnz(A))`.
pub fn coo_matvec<I: Integer, T: Scalar>(
    nnz: usize, /*npy_int64*/
    a_i: &[I],
    a_j: &[I],
    a_x: &[T],
    x_x: &[T],
    y_x: &mut [T],
) {
    for n in 0..nnz {
        let i = a_i[n].to_usize().unwrap();
        let j = a_j[n].to_usize().unwrap();
        y_x[i] += a_x[n] * x_x[j];
    }
}
