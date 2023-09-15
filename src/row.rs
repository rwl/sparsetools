use crate::traits::{Integer, Nonzero, Scalar};
use crate::util::axpy;
use std::cmp::min;
use std::collections::{HashMap, HashSet};

pub type Binop<T, T2> = fn(T, T) -> T2;

/// Extract k-th diagonal of CSR matrix A
///
/// # Input Arguments
/// ```txt
///   I  k             - diagonal to extract
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   T  Ax[n_col]     - nonzeros
/// ```
/// # Output Arguments
/// ```txt
///   T  Yx[min(n_row,n_col)] - diagonal entries
/// ```
/// # Notes
///
/// Output array Yx must be preallocated
///
/// Duplicate entries will be summed.
///
/// Complexity: Linear. Specifically O(nnz(A) + min(n_row,n_col))
pub fn csr_diagonal<I: Integer, T: Scalar>(
    k: isize,
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    y_x: &mut [T],
) {
    let first_row: usize = if k >= 0 { 0 } else { (-1 * k) as usize };
    let first_col: usize = if k >= 0 { k as usize } else { 0 };
    let n: usize = min(n_row - first_row, n_col - first_col);

    for i in 0..n {
        let row: usize = first_row + i;
        let col: usize = first_col + i;
        let row_begin: usize = a_p[row].to_usize().unwrap();
        let row_end: usize = a_p[row + 1].to_usize().unwrap();

        let mut diag: T = T::zero();
        for j in row_begin..row_end {
            if a_j[j].to_usize().unwrap() == col {
                diag += a_x[j];
            }
        }

        y_x[i] = diag;
    }
}

/// Expand a compressed row pointer into a row array
///
/// # Input Arguments
/// ```txt
///   I  n_row         - number of rows in A
///   I  Ap[n_row+1]   - row pointer
/// ```
/// # Output Arguments
/// ```txt
///   Bi  - row indices
/// ```
/// # Notes
///
/// Output array Bi must be preallocated
///
/// Complexity: Linear
pub fn expandptr<I: Integer>(n_row: usize, a_p: &[I], b_i: &mut [I]) {
    for i in 0..n_row {
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();
        for jj in start..end {
            b_i[jj] = I::from(i).unwrap();
        }
    }
}

/// Scale the rows of a CSR matrix *in place*
/// ```txt
///   A[i,:] *= X[i]
/// ```
pub fn csr_scale_rows<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    a_p: &[I],
    _a_j: &[I],
    a_x: &mut [T],
    x_x: &[T],
) {
    for i in 0..n_row {
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();
        for jj in start..end {
            a_x[jj] *= x_x[i];
        }
    }
}

/// Scale the columns of a CSR matrix *in place*
/// ```txt
///   A[:,i] *= X[i]
/// ```
pub fn csr_scale_columns<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &mut [T],
    x_x: &[T],
) {
    let nnz: usize = a_p[n_row].to_usize().unwrap();
    for i in 0..nnz {
        a_x[i] *= x_x[a_j[i].to_usize().unwrap()];
    }
}

/// Compute B += A for CSR matrix A, C-contiguous dense matrix B
///
/// # Input Arguments
/// ```txt
///   I  n_row           - number of rows in A
///   I  n_col           - number of columns in A
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - nonzero values
///   T  Bx[n_row*n_col] - dense matrix in row-major order
/// ```
pub fn csr_todense<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    // b_x: &mut [T],
    b_x: &mut Vec<Vec<T>>,
) {
    for i in 0..n_row {
        let row_start = a_p[i].to_usize().unwrap();
        let row_end = a_p[i + 1].to_usize().unwrap();
        for jj in row_start..row_end {
            let j = a_j[jj].to_usize().unwrap();
            b_x[i][j] += a_x[jj];
        }
    }
}

/// Determine whether the CSR column indices are in sorted order.
///
/// # Input Arguments
/// ```txt
///   I  n_row           - number of rows in A
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
/// ```
pub fn csr_has_sorted_indices<I: Integer>(n_row: usize, a_p: &[I], a_j: &[I]) -> bool {
    for i in 0..n_row {
        let start = a_p[i].to_usize().unwrap();
        let end = (a_p[i + 1] - I::one()).to_usize().unwrap();
        for jj in start..end {
            if a_j[jj] > a_j[jj + 1] {
                return false;
            }
        }
    }
    true
}

/// Determine whether the matrix structure is canonical CSR.
/// Canonical CSR implies that column indices within each row
/// are (1) sorted and (2) unique.  Matrices that meet these
/// conditions facilitate faster matrix computations.
///
/// # Input Arguments
/// ```txt
///   I  n_row           - number of rows in A
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
/// ```
pub fn csr_has_canonical_format<I: Integer>(n_row: usize, a_p: &[I], a_j: &[I]) -> bool {
    for i in 0..n_row {
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();
        if start > end {
            return false;
        }

        let start = (a_p[i] + I::one()).to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();
        for jj in start..end {
            if !(a_j[jj - 1] < a_j[jj]) {
                return false;
            }
        }
    }
    true
}

/// Sort CSR column indices inplace
///
/// # Input Arguments
/// ```txt
///   I  n_row           - number of rows in A
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - nonzeros
/// ```
pub fn csr_sort_indices<I: Integer, T: Scalar>(
    n_row: usize,
    a_p: &[I],
    a_j: &mut [I],
    a_x: &mut [T],
) {
    let mut temp: Vec<(I, T)> = vec![];

    for i in 0..n_row {
        let row_start = a_p[i].to_usize().unwrap();
        let row_end = a_p[i + 1].to_usize().unwrap();

        temp.resize(row_end - row_start, (I::zero(), T::zero()));
        let mut n = 0;
        for jj in row_start..row_end {
            temp[n].0 = a_j[jj];
            temp[n].1 = a_x[jj];
            n += 1;
        }

        temp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut n = 0;
        for jj in row_start..row_end {
            a_j[jj] = temp[n].0;
            a_x[jj] = temp[n].1;
            n += 1;
        }
    }
}

/// Compute B = A for CSR matrix A, CSC matrix B
///
/// Also, with the appropriate arguments can also be used to:
///
/// - compute B = A^t for CSR matrix A, CSR matrix B
/// - compute B = A^t for CSC matrix A, CSC matrix B
/// - convert CSC->CSR
///
/// # Input Arguments
/// ```txt
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   T  Ax[nnz(A)]    - nonzeros
/// ```
/// # Output Arguments
/// ```txt
///   I  Bp[n_col+1] - column pointer
///   I  Bi[nnz(A)]  - row indices
///   T  Bx[nnz(A)]  - nonzeros
/// ```
/// # Notes
///   
/// Output arrays Bp, Bi, Bx must be preallocated
///
/// Input: column indices *are not* assumed to be in sorted order
///
/// Output: row indices *will be* in sorted order
///
/// Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
pub fn csr_tocsc<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &mut [I],
    b_i: &mut [I],
    b_x: &mut [T],
) {
    let nnz: I = a_p[n_row];

    //compute number of non-zero entries per column of A
    // Bp.fill(Bp + n_col, I::zero());
    b_p.fill(I::zero());

    for n in 0..nnz.to_usize().unwrap() {
        b_p[a_j[n].to_usize().unwrap()] += I::one();
    }

    //cumsum the nnz per column to get Bp[]
    let mut cumsum = I::zero();
    for col in 0..n_col {
        let temp: I = b_p[col];
        b_p[col] = cumsum;
        cumsum += temp;
    }
    b_p[n_col] = nnz;

    for row in 0..n_row {
        let start = a_p[row].to_usize().unwrap();
        let end = a_p[row + 1].to_usize().unwrap();
        for jj in start..end {
            let col = a_j[jj].to_usize().unwrap();
            let dest = b_p[col].to_usize().unwrap();

            b_i[dest] = I::from(row).unwrap();
            b_x[dest] = a_x[jj];

            b_p[col] += I::one();
        }
    }

    let mut last = I::zero();
    for col in 0..=n_col {
        let temp: I = b_p[col];
        b_p[col] = last;
        last = temp;
    }
}

/// Compute C = A*B for CSR matrices A,B
///
/// # Input Arguments
/// ```txt
///   I  n_row       - number of rows in A
///   I  n_col       - number of columns in B (hence C is n_row by n_col)
///   I  Ap[n_row+1] - row pointer
///   I  Aj[nnz(A)]  - column indices
///   T  Ax[nnz(A)]  - nonzeros
///   I  Bp[?]       - row pointer
///   I  Bj[nnz(B)]  - column indices
///   T  Bx[nnz(B)]  - nonzeros
/// ```
/// # Output Arguments
/// ```txt
///   I  Cp[n_row+1] - row pointer
///   I  Cj[nnz(C)]  - column indices
///   T  Cx[nnz(C)]  - nonzeros
/// ```
/// # Notes
///
/// Output arrays Cp, Cj, and Cx must be preallocated.
/// In order to find the appropriate type for T, csr_matmat_maxnnz can be used
/// to find nnz(C).
///
/// Input:  A and B column indices *are not* assumed to be in sorted order
///
/// Output: C column indices *are not* assumed to be in sorted order
/// Cx will not contain any zero entries
///
/// Complexity: O(n_row*K^2 + max(n_row,n_col))
///             where K is the maximum nnz in a row of A
///             and column of B.
///
///
/// This is an implementation of the SMMP algorithm:
/// ```txt
///    "Sparse Matrix Multiplication Package (SMMP)"
///      Randolph E. Bank and Craig C. Douglas
///
///    http://citeseer.ist.psu.edu/445062.html
///    http://www.mgnet.org/~douglas/ccd-codes.html
/// ```txt

/// Compute the number of non-zeroes (nnz) in the result of C = A * B.
pub fn csr_matmat_maxnnz<I: Integer>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    b_p: &[I],
    b_j: &[I],
) -> usize /*npy_intp*/
{
    // method that uses O(n) temp storage
    // std::vector<I> mask(n_col, -1);
    let mut mask: Vec<isize> = vec![-1; n_col];

    let /*npy_intp*/ mut nnz = 0;
    for i in 0..n_row {
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();
        let /*npy_intp*/ mut row_nnz = 0;

        for jj in start..end {
            let j = a_j[jj].to_usize().unwrap();
            let b_start = b_p[j].to_usize().unwrap();
            let b_end = b_p[j + 1].to_usize().unwrap();

            for kk in b_start..b_end {
                let k = b_j[kk].to_usize().unwrap();
                if mask[k] != i as isize {
                    mask[k] = i as isize;
                    row_nnz += 1;
                }
            }
        }

        let /*npy_intp*/ next_nnz = nnz + row_nnz;

        if row_nnz > usize::MAX - nnz {
            // Index overflowed. Note that row_nnz <= n_col and cannot overflow
            panic!("nnz of the result is too large");
        }

        nnz = next_nnz;
    }

    nnz
}

/// Compute CSR entries for matrix C = A*B.
pub fn csr_matmat<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T],
) {
    // std::vector<I> next(n_col,-1);
    let mut next: Vec<isize> = vec![-1; n_col];
    // std::vector<T> sums(n_col, 0);
    let mut sums: Vec<T> = vec![T::zero(); n_col];

    let mut nnz = I::zero();

    c_p[0] = I::zero();

    for i in 0..n_row {
        let mut head: isize = -2;
        let mut length: I = I::zero();

        let jj_start = a_p[i].to_usize().unwrap();
        let jj_end = a_p[i + 1].to_usize().unwrap();
        for jj in jj_start..jj_end {
            let j = a_j[jj].to_usize().unwrap();
            let v = a_x[jj];

            let kk_start = b_p[j].to_usize().unwrap();
            let kk_end = b_p[j + 1].to_usize().unwrap();
            for kk in kk_start..kk_end {
                let k = b_j[kk].to_usize().unwrap();

                sums[k] += v * b_x[kk];

                if next[k] == -1 {
                    next[k] = head;
                    head = k as isize;
                    length += I::one();
                }
            }
        }

        for _jj in 0..length.to_usize().unwrap() {
            if sums[head as usize] != T::zero() {
                c_j[nnz.to_usize().unwrap()] = I::from(head).unwrap();
                c_x[nnz.to_usize().unwrap()] = sums[head as usize];
                nnz += I::one();
            }

            let temp = head as usize;
            head = next[head as usize];

            next[temp] = -1; //clear arrays
            sums[temp] = T::zero();
        }

        c_p[i + 1] = nnz;
    }
}

/// Compute C = A (binary_op) B for CSR matrices that are not
/// necessarily canonical CSR format.  Specifically, this method
/// works even when the input matrices have duplicate and/or
/// unsorted column indices within a given row.
///
/// Refer to [csr_binop_csr] for additional information
///
/// # Notes
///
/// Output arrays Cp, Cj, and Cx must be preallocated
/// If nnz(C) is not known a priori, a conservative bound is:
/// ```txt
///          nnz(C) <= nnz(A) + nnz(B)
/// ```
///
/// Input: A and B column indices are not assumed to be in sorted order
///
/// Output: C column indices are not generally in sorted order
///         C will not contain any duplicate entries or explicit zeros.
pub fn csr_binop_csr_general<I: Integer, T: Scalar, T2: Nonzero>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T2],
    op: Binop<T, T2>,
) {
    // Method that works for duplicate and/or unsorted indices

    // std::vector<I>  next(n_col,-1);
    let mut next: Vec<isize> = vec![-1; n_col];
    // std::vector<T> A_row(n_col, 0);
    let mut a_row: Vec<T> = vec![T::zero(); n_col];
    // std::vector<T> B_row(n_col, 0);
    let mut b_row: Vec<T> = vec![T::zero(); n_col];

    let mut nnz: I = I::zero();
    c_p[0] = I::zero();

    for i in 0..n_row {
        let mut head: isize = -2;
        let mut length: I = I::zero();

        //add a row of A to A_row
        let i_start = a_p[i].to_usize().unwrap();
        let i_end = a_p[i + 1].to_usize().unwrap();
        for jj in i_start..i_end {
            let j = a_j[jj].to_usize().unwrap();

            a_row[j] += a_x[jj];

            if next[j] == -1 {
                next[j] = head;
                head = j as isize;
                length += I::one();
            }
        }

        //add a row of B to B_row
        let i_start = b_p[i].to_usize().unwrap();
        let i_end = b_p[i + 1].to_usize().unwrap();
        for jj in i_start..i_end {
            let j = b_j[jj].to_usize().unwrap();

            b_row[j] += b_x[jj];

            if next[j] == -1 {
                next[j] = head;
                head = j as isize;
                length += I::one();
            }
        }

        // scan through columns where A or B has
        // contributed a non-zero entry
        for _jj in 0..length.to_usize().unwrap() {
            let result: T2 = op(a_row[head as usize], b_row[head as usize]);

            if result.nonzero() {
                c_j[nnz.to_usize().unwrap()] = I::from(head).unwrap();
                c_x[nnz.to_usize().unwrap()] = result;
                nnz += I::one();
            }

            let temp = head as usize;
            head = next[head as usize];

            next[temp] = -1;
            a_row[temp] = T::zero();
            b_row[temp] = T::zero();
        }

        c_p[i + 1] = nnz;
    }
}

/// Compute C = A (binary_op) B for CSR matrices that are in the
/// canonical CSR format.  Specifically, this method requires that
/// the rows of the input matrices are free of duplicate column indices
/// and that the column indices are in sorted order.
///
/// Refer to [csr_binop_csr] for additional information
///
/// # Notes
///
/// Input:  A and B column indices are assumed to be in sorted order
///
/// Output: C column indices will be in sorted order
///         Cx will not contain any zero entries
pub fn csr_binop_csr_canonical<I: Integer, T: Scalar, T2: Nonzero>(
    n_row: usize,
    _n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T2],
    op: Binop<T, T2>,
) {
    // Method that works for canonical CSR matrices

    c_p[0] = I::zero();
    let mut nnz: usize = 0;

    for i in 0..n_row {
        let mut a_pos = a_p[i].to_usize().unwrap();
        let mut b_pos = b_p[i].to_usize().unwrap();
        let a_end = a_p[i + 1].to_usize().unwrap();
        let b_end = b_p[i + 1].to_usize().unwrap();

        //while not finished with either row
        while a_pos < a_end && b_pos < b_end {
            let aj: I = a_j[a_pos];
            let bj: I = b_j[b_pos];

            if aj == bj {
                let result: T2 = op(a_x[a_pos], b_x[b_pos]);
                if result.nonzero() {
                    c_j[nnz] = aj;
                    c_x[nnz] = result;
                    nnz += 1;
                }
                a_pos += 1;
                b_pos += 1;
            } else if aj < bj {
                let result: T2 = op(a_x[a_pos], T::zero());
                if result.nonzero() {
                    c_j[nnz] = aj;
                    c_x[nnz] = result;
                    nnz += 1;
                }
                a_pos += 1;
            } else {
                //B_j < A_j
                let result: T2 = op(T::zero(), b_x[b_pos]);
                if result.nonzero() {
                    c_j[nnz] = bj;
                    c_x[nnz] = result;
                    nnz += 1;
                }
                b_pos += 1;
            }
        }

        //tail
        while a_pos < a_end {
            let result: T2 = op(a_x[a_pos], T::zero());
            if result.nonzero() {
                c_j[nnz] = a_j[a_pos];
                c_x[nnz] = result;
                nnz += 1;
            }
            a_pos += 1;
        }
        while b_pos < b_end {
            let result: T2 = op(T::zero(), b_x[b_pos]);
            if result.nonzero() {
                c_j[nnz] = b_j[b_pos];
                c_x[nnz] = result;
                nnz += 1;
            }
            b_pos += 1;
        }

        c_p[i + 1] = I::from(nnz).unwrap();
    }
}

/// Compute C = A (binary_op) B for CSR matrices A,B where the column
/// indices with the rows of A and B are known to be sorted.
///
/// # Input Arguments
/// ```txt
///   I    n_row       - number of rows in A (and B)
///   I    n_col       - number of columns in A (and B)
///   I    Ap[n_row+1] - row pointer
///   I    Aj[nnz(A)]  - column indices
///   T    Ax[nnz(A)]  - nonzeros
///   I    Bp[n_row+1] - row pointer
///   I    Bj[nnz(B)]  - column indices
///   T    Bx[nnz(B)]  - nonzeros
/// ```
/// # Output Arguments
/// ```txt
///   I    Cp[n_row+1] - row pointer
///   I    Cj[nnz(C)]  - column indices
///   T    Cx[nnz(C)]  - nonzeros
/// ```
///
/// # Notes
///
/// Output arrays Cp, Cj, and Cx must be preallocated
/// If nnz(C) is not known a priori, a conservative bound is:
/// ```txt
///          nnz(C) <= nnz(A) + nnz(B)
/// ```
///
/// Input: A and B column indices are not assumed to be in sorted order.
///
/// Output: C column indices will be in sorted if both A and B have sorted indices.
///         Cx will not contain any zero entries
pub fn csr_binop_csr<I: Integer, T: Scalar, T2: Nonzero>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T2],
    op: Binop<T, T2>,
) {
    if csr_has_canonical_format(n_row, a_p, a_j) && csr_has_canonical_format(n_row, b_p, b_j) {
        csr_binop_csr_canonical(
            n_row, n_col, a_p, a_j, a_x, b_p, b_j, b_x, c_p, c_j, c_x, op,
        );
    } else {
        csr_binop_csr_general(
            n_row, n_col, a_p, a_j, a_x, b_p, b_j, b_x, c_p, c_j, c_x, op,
        );
    }
}

// element-wise binary operations //

pub fn csr_ne_csr<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [bool],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a != b,
    );
}

pub fn csr_eq_csr<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [bool],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a == b,
    );
}

pub fn csr_lt_csr<I: Integer, T: Scalar + PartialOrd>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [bool],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a < b,
    );
}

pub fn csr_le_csr<I: Integer, T: Scalar + PartialOrd>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [bool],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a <= b,
    );
}

pub fn csr_gt_csr<I: Integer, T: Scalar + PartialOrd>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [bool],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a > b,
    );
}

pub fn csr_ge_csr<I: Integer, T: Scalar + PartialOrd>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [bool],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a >= b,
    );
}

pub fn csr_mul_csr<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a * b,
    );
}

pub fn csr_div_csr<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a / b,
    );
}

pub fn csr_add_csr<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a + b,
    );
}

pub fn csr_sub_csr<I: Integer, T: Scalar>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| a - b,
    );
}

pub fn csr_max_csr<I: Integer, T: Scalar + PartialOrd>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| {
            if a > b {
                a
            } else {
                b
            }
        },
    );
}
pub fn csr_min_csr<I: Integer, T: Scalar + PartialOrd>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_p: &[I],
    b_j: &[I],
    b_x: &[T],
    c_p: &mut [I],
    c_j: &mut [I],
    c_x: &mut [T],
) {
    csr_binop_csr(
        n_row,
        n_col,
        a_p,
        a_j,
        a_x,
        b_p,
        b_j,
        b_x,
        c_p,
        c_j,
        c_x,
        |a, b| {
            if a < b {
                a
            } else {
                b
            }
        },
    );
}

/// Sum together duplicate column entries in each row of CSR matrix A
///
/// # Input Arguments
/// ```txt
///   I    n_row       - number of rows in A (and B)
///   I    n_col       - number of columns in A (and B)
///   I    Ap[n_row+1] - row pointer
///   I    Aj[nnz(A)]  - column indices
///   T    Ax[nnz(A)]  - nonzeros
/// ```
///
/// # Notes
///
/// The column indices within each row must be in sorted order.
/// Explicit zeros are retained.
pub fn csr_sum_duplicates<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    a_p: &mut [I],
    a_j: &mut [I],
    a_x: &mut [T],
) {
    let mut nnz: usize = 0;
    let mut row_end: usize = 0;
    for i in 0..n_row {
        let mut jj = row_end;
        row_end = a_p[i + 1].to_usize().unwrap();
        while jj < row_end {
            let j: I = a_j[jj];
            let mut x: T = a_x[jj];
            jj += 1;
            while jj < row_end && a_j[jj] == j {
                x += a_x[jj];
                jj += 1;
            }
            a_j[nnz] = j;
            a_x[nnz] = x;
            nnz += 1;
        }
        a_p[i + 1] = I::from(nnz).unwrap();
    }
}

/// Eliminate zero entries from CSR matrix A
///
/// # Input Arguments
/// ```txt
///   I    n_row       - number of rows in A (and B)
///   I    n_col       - number of columns in A (and B)
///   I    Ap[n_row+1] - row pointer
///   I    Aj[nnz(A)]  - column indices
///   T    Ax[nnz(A)]  - nonzeros
/// ```
pub fn csr_eliminate_zeros<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    a_p: &mut [I],
    a_j: &mut [I],
    a_x: &mut [T],
) {
    let mut nnz: usize = 0;
    let mut row_end: usize = 0;
    for i in 0..n_row {
        let mut jj = row_end;
        row_end = a_p[i + 1].to_usize().unwrap();
        while jj < row_end {
            let j: I = a_j[jj];
            let x: T = a_x[jj];
            if x != T::zero() {
                a_j[nnz] = j;
                a_x[nnz] = x;
                nnz += 1;
            }
            jj += 1;
        }
        a_p[i + 1] = I::from(nnz).unwrap();
    }
}

/// Compute Y += A*X for CSR matrix A and dense vectors X,Y
///
/// # Input Arguments
/// ```txt
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   T  Ax[nnz(A)]    - nonzeros
///   T  Xx[n_col]     - input vector
/// ```
/// # Output Arguments
/// ```txt
///   T  Yx[n_row]     - output vector
/// ```
/// # Note:
///   
/// Output array Yx must be preallocated
///
/// Complexity: Linear. Specifically O(nnz(A) + n_row)
pub fn csr_matvec<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    x_x: &[T],
    y_x: &mut [T],
) {
    for i in 0..n_row {
        let mut sum: T = y_x[i];
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();
        for jj in start..end {
            sum += a_x[jj] * x_x[a_j[jj].to_usize().unwrap()];
        }
        y_x[i] = sum;
    }
}

/// Compute Y += A*X for CSR matrix A and dense block vectors X,Y
///
///
/// # Input Arguments
/// ```txt
///   I  n_row            - number of rows in A
///   I  n_col            - number of columns in A
///   I  n_vecs           - number of column vectors in X and Y
///   I  Ap[n_row+1]      - row pointer
///   I  Aj[nnz(A)]       - column indices
///   T  Ax[nnz(A)]       - nonzeros
///   T  Xx[n_col,n_vecs] - input vector
/// ```
/// # Output Arguments
/// ```txt
///   T  Yx[n_row,n_vecs] - output vector
/// ```
pub fn csr_matvecs<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    n_vecs: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    x_x: &[T],
    y_x: &mut [T],
) {
    for i in 0..n_row {
        let y = &mut y_x[n_vecs * i..];

        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();
        for jj in start..end {
            let j = a_j[jj].to_usize().unwrap();
            let a: T = a_x[jj];
            let x = &x_x[n_vecs * j..];
            axpy(n_vecs, a, x, y);
        }
    }
}

pub fn csr_select<I: Integer, T: Scalar>(
    _n_row: usize,
    _n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    rowidx: &[I],
    colidx: &[I],
    b_p: &mut Vec<I>,
    b_j: &mut Vec<I>,
    b_x: &mut Vec<T>,
) {
    let new_n_row = rowidx.len();
    let mut new_nnz = 0;

    // Count nonzeros total/per row.
    for row in rowidx {
        let urow = row.to_usize().unwrap();
        let row_start = a_p[urow].to_usize().unwrap();
        let row_end = a_p[urow + 1].to_usize().unwrap();

        let mut colptrs = HashSet::new(); // TODO: Cache for second pass?

        for jj in row_start..row_end {
            colptrs.insert(a_j[jj].to_usize().unwrap());
        }
        for col in colidx {
            if colptrs.contains(&col.to_usize().unwrap()) {
                new_nnz += 1;
            }
        }
    }

    // Allocate.
    b_p.resize(new_n_row + 1, I::zero());
    b_j.resize(new_nnz, I::zero());
    b_x.resize(new_nnz, T::zero());

    // Assign.
    b_p[0] = I::zero();
    let mut kk = 0;
    for (r, row) in rowidx.iter().enumerate() {
        let urow = row.to_usize().unwrap();
        let start = a_p[urow].to_usize().unwrap();
        let end = a_p[urow + 1].to_usize().unwrap();

        let mut colptrs = HashMap::new();
        for jj in start..end {
            colptrs.insert(a_j[jj].to_usize().unwrap(), jj);
        }

        for (c, col) in colidx.iter().enumerate() {
            if let Some(&jj) = colptrs.get(&col.to_usize().unwrap()) {
                b_j[kk] = I::from(c).unwrap();
                b_x[kk] = a_x[jj];
                kk += 1;
            }
        }
        b_p[r + 1] = I::from(kk).unwrap();
    }
}

pub fn get_csr_submatrix<I: Integer, T: Scalar>(
    _n_row: usize,
    _n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    ir0: I,
    ir1: I,
    ic0: I,
    ic1: I,
    b_p: &mut Vec<I>,
    b_j: &mut Vec<I>,
    b_x: &mut Vec<T>,
) {
    let uir0 = ir0.to_usize().unwrap();

    let new_n_row: usize = (ir1 - ir0).to_usize().unwrap();
    //I new_n_col = ic1 - ic0;  //currently unused
    let mut new_nnz = 0;
    let mut kk = 0;

    // Count nonzeros total/per row.
    for i in 0..new_n_row {
        let row_start = a_p[uir0 + i].to_usize().unwrap();
        let row_end = a_p[uir0 + i + 1].to_usize().unwrap();

        for jj in row_start..row_end {
            if (a_j[jj] >= ic0) && (a_j[jj] < ic1) {
                new_nnz += 1;
            }
        }
    }

    // Allocate.
    b_p.resize(new_n_row + 1, I::zero());
    b_j.resize(new_nnz, I::zero());
    b_x.resize(new_nnz, T::zero());

    // Assign.
    // (*Bp)[0] = 0;
    b_p[0] = I::zero();
    for i in 0..new_n_row {
        let row_start = a_p[uir0 + i].to_usize().unwrap();
        let row_end = a_p[uir0 + i + 1].to_usize().unwrap();

        for jj in row_start..row_end {
            if (a_j[jj] >= ic0) && (a_j[jj] < ic1) {
                // (*Bj)[kk] = Aj[jj] - ic0;
                b_j[kk] = a_j[jj] - ic0;
                // (*Bx)[kk] = Ax[jj];
                b_x[kk] = a_x[jj];
                kk += 1;
            }
        }
        // (*Bp)[i+1] = kk;
        b_p[i + 1] = I::from(kk).unwrap();
    }
}

/// Slice rows given as an array of indices.
///
/// # Input Arguments
/// ```txt
///   I  n_row_idx       - number of row indices
///   I  rows[n_row_idx] - row indices for indexing
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - data
/// ```
/// # Output Arguments
/// ```txt
///   I  Bj - new column indices
///   T  Bx - new data
/// ```
pub fn csr_row_index<I: Integer, T: Scalar>(
    n_row_idx: usize,
    rows: &[usize],
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_j: &mut Vec<I>,
    b_x: &mut Vec<T>,
) {
    for i in 0..n_row_idx {
        let row = rows[i];

        let row_start = a_p[row].to_usize().unwrap();
        let row_end = a_p[row + 1].to_usize().unwrap();

        // Bj = copy(Aj + row_start, Aj + row_end, Bj);
        b_j.extend_from_slice(&a_j[row_start..row_end]);
        // Bx = copy(Ax + row_start, Ax + row_end, Bx);
        b_x.extend_from_slice(&a_x[row_start..row_end]);
    }
}

/// Slice rows given as a (start, stop, step) tuple.
///
/// # Input Arguments
/// ```txt
///   I  start
///   I  stop
///   I  step
///   I  Ap[N+1]    - row pointer
///   I  Aj[nnz(A)] - column indices
///   T  Ax[nnz(A)] - data
/// ```
/// # Output Arguments
/// ```txt
///   I  Bj - new column indices
///   T  Bx - new data
/// ```
pub fn csr_row_slice<I: Integer, T: Scalar>(
    start: usize,
    stop: usize,
    step: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    b_j: &mut Vec<I>,
    b_x: &mut Vec<T>,
) {
    if step > 0 {
        let mut row = start;
        while row < stop {
            let row_start = a_p[row].to_usize().unwrap();
            let row_end = a_p[row + 1].to_usize().unwrap();

            // Bj = copy(Aj + row_start, Aj + row_end, Bj);
            b_j.extend_from_slice(&a_j[row_start..row_end]);
            // Bx = copy(Ax + row_start, Ax + row_end, Bx);
            b_x.extend_from_slice(&a_x[row_start..row_end]);

            row += step;
        }
    } else {
        let mut row = start;
        while row > stop {
            let row_start = a_p[row].to_usize().unwrap();
            let row_end = a_p[row + 1].to_usize().unwrap();

            // Bj = copy(Aj + row_start, Aj + row_end, Bj);
            b_j.extend_from_slice(&a_j[row_start..row_end]);
            // Bx = copy(Ax + row_start, Ax + row_end, Bx);
            b_x.extend_from_slice(&a_x[row_start..row_end]);

            row += step;
        }
    }
}

/// Slice columns given as an array of indices (pass 1).
/// This pass counts idx entries and computes a new indptr.
///
/// # Input Arguments
/// ```txt
///   I  n_idx           - number of indices to slice
///   I  col_idxs[n_idx] - indices to slice
///   I  n_row           - major axis dimension
///   I  n_col           - minor axis dimension
///   I  Ap[n_row+1]     - indptr
///   I  Aj[nnz(A)]      - indices
/// ```
/// # Output Arguments
/// ```txt
///   I  col_offsets[n_col] - cumsum of index repeats
///   I  Bp[n_row+1]        - new indptr
/// ```
pub fn csr_column_index1<I: Integer>(
    n_idx: usize,
    col_idxs: &[usize],
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    col_offsets: &mut [I],
    b_p: &mut [I],
) {
    // bincount(col_idxs)
    for jj in 0..n_idx {
        let j = col_idxs[jj];
        col_offsets[j] += I::one();
    }

    // Compute new indptr
    let mut new_nnz = I::zero();
    b_p[0] = I::zero();
    for i in 0..n_row {
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();

        for jj in start..end {
            new_nnz += col_offsets[a_j[jj].to_usize().unwrap()];
        }
        b_p[i + 1] = new_nnz;
    }

    // cumsum in-place
    for j in 1..n_col {
        col_offsets[j] += col_offsets[j - 1];
    }
}

/// Slice columns given as an array of indices (pass 2).
/// This pass populates indices/data entries for selected columns.
///
/// # Input Arguments
/// ```txt
///   I  col_order[n_idx]   - order of col indices
///   I  col_offsets[n_col] - cumsum of col index counts
///   I  nnz                - nnz(A)
///   I  Aj[nnz(A)]         - column indices
///   T  Ax[nnz(A)]         - data
/// ```
///
/// # Output Arguments
/// ```txt
///   I  Bj[nnz(B)] - new column indices
///   T  Bx[nnz(B)] - new data
/// ```
pub fn csr_column_index2<I: Integer, T: Scalar>(
    col_order: &[I],
    col_offsets: &[I],
    nnz: usize,
    a_j: &[I],
    a_x: &[T],
    b_j: &mut [I],
    b_x: &mut [T],
) {
    let mut n: usize = 0;
    for jj in 0..nnz {
        let j = a_j[jj].to_usize().unwrap();
        let offset = col_offsets[j].to_usize().unwrap();
        let prev_offset: usize = if j == 0 {
            0
        } else {
            col_offsets[j - 1].to_usize().unwrap()
        };
        if offset != prev_offset {
            let v: T = a_x[jj];
            for k in prev_offset..offset {
                b_j[n] = col_order[k];
                b_x[n] = v;
                n += 1;
            }
        }
    }
}

/// Count the number of occupied diagonals in CSR matrix A
///
/// # Input Arguments
///   I  nnz             - number of nonzeros in A
///   I  Ai[nnz(A)]      - row indices
///   I  Aj[nnz(A)]      - column indices
pub fn csr_count_diagonals<I: Integer>(n_row: usize, a_p: &[I], a_j: &[I]) -> usize {
    let mut diagonals = HashSet::<usize>::new();

    for i in 0..n_row {
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();

        for jj in start..end {
            diagonals.insert(a_j[jj].to_usize().unwrap() - i);
        }
    }
    diagonals.len()
}

/*
/// Stack CSR matrices in A horizontally (column wise)
///
/// # Input Arguments
///   I  n_blocks                      - number of matrices in A
///   I  n_row                         - number of rows in any matrix in A
///   I  n_col_cat[n_blocks]           - number of columns in each matrix in A concatenated
///   I  Ap_cat[n_blocks*(n_row + 1)]  - row indices of each matrix in A concatenated
///   I  Aj_cat[nnz(A)]                - column indices of each matrix in A concatenated
///   T  Ax_cat[nnz(A)]                - nonzeros of each matrix in A concatenated
///
/// # Output Arguments
///   I Bp  - row pointer
///   I Bj  - column indices
///   T Bx  - nonzeros
///
/// Note:
///   All output arrays Bp, Bj, Bx must be preallocated
///
///   Complexity: Linear.  Specifically O(nnz(A) + n_blocks)
pub fn csr_hstack<I: Integer + FromPrimitive, T: Scalar>(
    n_blocks: I,
    n_row: I,
    n_col_cat: &[I],
    bAp: &[&[I]],
    bAj: &[&[I]],
    bAx: &[&[T]],
    // Ap_cat: &[&[I]],
    // Aj_cat: &[&[I]],
    // Ax_cat: &[&[T]],
    Bp: &mut [I],
    Bj: &mut [I],
    Bx: &mut [T],
) {
    // First, mark the blocks in the input data while
    // computing their column offsets:
    let mut col_offset: Vec<I> = vec![I::zero(); n_blocks.to_usize().unwrap()];
    // std::vector<const I*> bAp(n_blocks);
    // let mut bAp = vec![n_blocks];
    // std::vector<const I*> bAj(n_blocks);
    // let mut bAj = vec![n_blocks];
    // std::vector<const T*> bAx(n_blocks);
    // let mut bAx = vec![n_blocks];
    col_offset[0] = I::zero();
    // bAp[0] = Ap_cat;
    // bAj[0] = Aj_cat;
    // bAx[0] = Ax_cat;
    for b in 1..n_blocks.to_usize().unwrap() {
        col_offset[b] = col_offset[b - 1] + n_col_cat[b - 1];
        // bAp[b] = bAp[b - 1] + (n_row + 1);
        // bAj[b] = bAj[b - 1] + bAp[b - 1][n_row];
        // bAx[b] = bAx[b - 1] + bAp[b - 1][n_row];
    }

    // Next, build the full output matrix:
    Bp[0] = I::zero();
    let mut s: usize = 0;
    for i in 0..n_row.to_usize().unwrap() {
        for b in 0..n_blocks.to_usize().unwrap() {
            let jj_start = bAp[b][i].to_usize().unwrap();
            let jj_end = bAp[b][i + 1].to_usize().unwrap();
            let offset = col_offset[b];
            // FIXME
            // for x in bAj[b][jj_start]..bAj[b][jj_end] {
            //     Bj[s] += x + offset;
            // }
            // for x in bAx[b][jj_start]..bAx[b][jj_end] {
            //     Bx[s] = x;
            // }
            // transform(&bAj[b][jj_start], &bAj[b][jj_end], &Bj[s], [&](I x){return (x + offset);});
            // copy(&bAx[b][jj_start], &bAx[b][jj_end], &Bx[s]);
            s += jj_end - jj_start;
        }
        Bp[i + 1] = I::from_usize(s).unwrap();
    }
}
*/

pub fn csr_tocoo<I: Integer, T: Scalar>(
    n_row: usize,
    _n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    a_i: &mut [I],
    b_j: &mut [I],
    b_x: &mut [T],
) {
    let mut k = 0;
    for i in 0..n_row {
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();
        for jj in start..end {
            a_i[k] = I::from(i).unwrap();
            b_j[k] = a_j[jj];
            b_x[k] = a_x[jj];
            k += 1;
        }
    }
}
