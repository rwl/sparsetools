use crate::dense::axpy;
use crate::scalar::Scalar;
use crate::util::Binop;
use num_traits::{AsPrimitive, FromPrimitive, NumCast, PrimInt, Signed, ToPrimitive};
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::ops::MulAssign;
use superslice::Ext;

/// Extract k-th diagonal of CSR matrix A
///
/// Input Arguments:
///   I  k             - diagonal to extract
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   T  Ax[n_col]     - nonzeros
///
/// Output Arguments:
///   T  Yx[min(n_row,n_col)] - diagonal entries
///
/// Note:
///   Output array Yx must be preallocated
///
///   Duplicate entries will be summed.
///
///   Complexity: Linear.  Specifically O(nnz(A) + min(n_row,n_col))
pub fn csr_diagonal<I: PrimInt + Signed, T: Scalar>(
    k: isize,
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Yx: &mut [T],
) {
    let first_row: usize = if k >= 0 { 0 } else { (-1 * k) as usize };
    let first_col: usize = if k >= 0 { k as usize } else { 0 };
    let N: usize = min(
        n_row.to_usize().unwrap() - first_row,
        n_col.to_usize().unwrap() - first_col,
    );

    for i in 0..N {
        let row: usize = first_row + i;
        let col: usize = first_col + i;
        let row_begin: usize = Ap[row].to_usize().unwrap();
        let row_end: usize = Ap[row + 1].to_usize().unwrap();

        let mut diag: T = T::zero();
        for j in row_begin..row_end {
            if Aj[j].to_usize().unwrap() == col {
                diag = diag + Ax[j];
            }
        }

        Yx[i] = diag;
    }
}

/// Expand a compressed row pointer into a row array
///
/// Input Arguments:
///   I  n_row         - number of rows in A
///   I  Ap[n_row+1]   - row pointer
///
/// Output Arguments:
///   Bi  - row indices
///
/// Note:
///   Output array Bi must be preallocated
///
/// Note:
///   Complexity: Linear
pub fn expandptr<I: PrimInt>(n_row: I, Ap: &[I], Bi: &mut [I]) {
    for i in 0..n_row.to_usize().unwrap() {
        let start = Ap[i].to_usize().unwrap();
        let end = Ap[i + 1].to_usize().unwrap();
        for jj in start..end {
            Bi[jj] = I::from(i).unwrap();
        }
    }
}

/// Scale the rows of a CSR matrix *in place*
///
///   A[i,:] *= X[i]
pub fn csr_scale_rows<I: PrimInt, T: Scalar + MulAssign>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &mut [T],
    Xx: &[T],
) {
    for i in 0..n_row.to_usize().unwrap() {
        let start = Ap[i].to_usize().unwrap();
        let end = Ap[i + 1].to_usize().unwrap();
        for jj in start..end {
            Ax[jj] *= Xx[i];
        }
    }
}

/// Scale the columns of a CSR matrix *in place*
///
///   A[:,i] *= X[i]
pub fn csr_scale_columns<I: PrimInt, T: Scalar + MulAssign>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &mut [T],
    Xx: &[T],
) {
    let nnz: usize = Ap[n_row.to_usize().unwrap()].to_usize().unwrap();
    for i in 0..nnz {
        Ax[i] *= Xx[Aj[i].to_usize().unwrap()];
    }
}

/// Compute the number of occupied RxC blocks in a matrix
///
/// Input Arguments:
///   I  n_row         - number of rows in A
///   I  R             - row blocksize
///   I  C             - column blocksize
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///
/// Output Arguments:
///   I  num_blocks    - number of blocks
///
/// Note:
///   Complexity: Linear
pub fn csr_count_blocks<I: PrimInt + Signed>(
    n_row: I,
    n_col: I,
    R: I,
    C: I,
    Ap: &[I],
    Aj: &[I],
) -> I {
    // std::vector<I> mask(n_col/C + 1,-1);
    let mask = vec![-I::one(); (n_col / C + I::one()).to_usize().unwrap()];
    let mut n_blks: I = I::zero();
    for i in 0..n_row.to_usize().unwrap() {
        let bi: I = i / R;
        for jj in Ap[i]..Ap[i + 1] {
            let bj: I = Aj[jj] / C;
            if mask[bj] != bi {
                mask[bj] = bi;
                n_blks += 1;
            }
        }
    }
    n_blks
}

/// Convert a CSR matrix to BSR format
///
/// Input Arguments:
///   I  n_row           - number of rows in A
///   I  n_col           - number of columns in A
///   I  R               - row blocksize
///   I  C               - column blocksize
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - nonzero values
///
/// Output Arguments:
///   I  Bp[n_row/R + 1] - block row pointer
///   I  Bj[nnz(B)]      - column indices
///   T  Bx[nnz(B)]      - nonzero blocks
///
/// Note:
///   Complexity: Linear
///   Output arrays must be preallocated (with Bx initialized to zero)
pub fn csr_tobsr<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    R: I,
    C: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bp: &mut [I],
    Bj: &mut [I],
    Bx: &mut [T],
) {
    // std::vector<T*> blocks(n_col/C + 1, (T*)0 );
    let mut blocks = vec![T::zero(); n_col / C + 1];

    assert!(n_row % R == 0);
    assert!(n_col % C == 0);

    let n_brow: I = n_row / R;
    //let n_bcol: I = n_col / C;

    let RC: I = R * C;
    let mut n_blks: I = I::zero();

    Bp[0] = 0;

    for bi in 0..n_brow {
        for r in 0..R {
            let i: I = R * bi + r; //row index
            for jj in Ap[i]..Ap[i + 1] {
                let j: I = Aj[jj]; //column index

                let bj: I = j / C;
                let c: I = j % C;

                if blocks[bj] == 0 {
                    blocks[bj] = Bx + RC * n_blks; // FIXME
                    Bj[n_blks] = bj;
                    n_blks += 1;
                }

                *(blocks[bj] + C * r + c) += Ax[jj];
            }
        }

        for jj in Ap[R * bi]..Ap[R * (bi + 1)] {
            blocks[Aj[jj] / C] = T::zero();
        }

        Bp[bi + 1] = n_blks;
    }
}

/// Compute B += A for CSR matrix A, C-contiguous dense matrix B
///
/// Input Arguments:
///   I  n_row           - number of rows in A
///   I  n_col           - number of columns in A
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - nonzero values
///   T  Bx[n_row*n_col] - dense matrix in row-major order
pub fn csr_todense<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bx: &mut [T],
) {
    let mut Bx_row/*: *T */= Bx;
    for i in 0..n_row {
        for jj in Ap[i]..Ap[i + 1] {
            Bx_row[Aj[jj]] += Ax[jj];
        }
        Bx_row += n_col /*as npy_intp*/;
    }
}

/// Determine whether the CSR column indices are in sorted order.
///
/// Input Arguments:
///   I  n_row           - number of rows in A
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
pub fn csr_has_sorted_indices<I: PrimInt>(n_row: I, Ap: &[I], Aj: &[I]) -> bool {
    for i in 0..n_row {
        for jj in Ap[i]..Ap[i + 1] - 1 {
            if Aj[jj] > Aj[jj + 1] {
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
/// Input Arguments:
///   I  n_row           - number of rows in A
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
pub fn csr_has_canonical_format<I: PrimInt>(n_row: I, Ap: &[I], Aj: &[I]) -> bool {
    for i in 0..n_row {
        if Ap[i] > Ap[i + 1] {
            return false;
        }
        for jj in Ap[i] + 1..Ap[i + 1] {
            if !(Aj[jj - 1] < Aj[jj]) {
                return false;
            }
        }
    }
    return true;
}

/// Sort CSR column indices inplace
///
/// Input Arguments:
///   I  n_row           - number of rows in A
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - nonzeros
pub fn csr_sort_indices<I: PrimInt, T: Scalar>(n_row: I, Ap: &[I], Aj: &[I], Ax: &[T]) {
    // std::vector< std::pair<I,T> > temp;
    let mut temp: Vec<(I, T)> = vec![];

    for i in 0..n_row {
        let row_start: I = Ap[i];
        let row_end: I = Ap[i + 1];

        temp.resize(row_end - row_start, (I::zero(), T::zero()));
        let mut n = 0;
        for jj in row_start..row_end {
            temp[n].first = Aj[jj];
            temp[n].second = Ax[jj];
            n += 1;
        }

        // std::sort(temp.begin(),temp.end(),kv_pair_less<I,T>);
        temp.sort();

        let mut n = 0;
        for jj in row_start..row_end {
            Aj[jj] = temp[n].first;
            Ax[jj] = temp[n].second;
        }
        n += 1;
    }
}

/// Compute B = A for CSR matrix A, CSC matrix B
///
/// Also, with the appropriate arguments can also be used to:
///   - compute B = A^t for CSR matrix A, CSR matrix B
///   - compute B = A^t for CSC matrix A, CSC matrix B
///   - convert CSC->CSR
///
/// Input Arguments:
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   T  Ax[nnz(A)]    - nonzeros
///
/// Output Arguments:
///   I  Bp[n_col+1] - column pointer
///   I  Bi[nnz(A)]  - row indices
///   T  Bx[nnz(A)]  - nonzeros
///
/// Note:
///   Output arrays Bp, Bi, Bx must be preallocated
///
/// Note:
///   Input:  column indices *are not* assumed to be in sorted order
///   Output: row indices *will be* in sorted order
///
///   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
pub fn csr_tocsc<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bp: &mut [I],
    Bi: &mut [I],
    Bx: &mut [T],
) {
    let nnz: I = Ap[n_row];

    //compute number of non-zero entries per column of A
    // Bp.fill(Bp + n_col, I::zero());
    Bp.fill(I::zero());

    for n in 0..nnz {
        Bp[Aj[n]] += 1;
    }

    //cumsum the nnz per column to get Bp[]
    let mut cumsum = 0;
    for col in 0..n_col {
        let temp: I = Bp[col];
        Bp[col] = cumsum;
        cumsum += temp;
    }
    Bp[n_col] = nnz;

    for row in 0..n_row {
        for jj in Ap[row]..Ap[row + 1] {
            let col: I = Aj[jj];
            let dest: I = Bp[col];

            Bi[dest] = row;
            Bx[dest] = Ax[jj];

            Bp[col] += 1;
        }
    }

    let mut last = 0;
    for col in 0..=n_col {
        let temp: I = Bp[col];
        Bp[col] = last;
        last = temp;
    }
}

/// Compute B = A for CSR matrix A, ELL matrix B
///
/// Input Arguments:
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   T  Ax[nnz(A)]    - nonzeros
///   I  row_length    - maximum nnz in a row of A
///
/// Output Arguments:
///   I  Bj[n_row * row_length]  - column indices
///   T  Bx[n_row * row_length]  - nonzeros
///
/// Note:
///   Output arrays Bj, Bx must be preallocated
///   Duplicate entries in A are not merged.
///   Explicit zeros in A are carried over to B.
///   Rows with fewer than row_length columns are padded with zeros.
pub fn csr_toell<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    row_length: I,
    Bj: &mut [I],
    Bx: &mut [T],
) {
    // const npy_intp ell_nnz = (npy_intp)row_length * n_row;
    let ell_nnz = row_length * n_row;
    // Bj.fill(Bj + ell_nnz, 0);
    Bj.fill(I::zero());
    // Bx.fill(Bx + ell_nnz, 0);
    Bx.fill(T::zero());

    for i in 0..n_row {
        let Bj_row/*: *I */= Bj + /*(npy_intp)*/row_length * i;
        let Bx_row/*: *T */= Bx + row_length * i;
        for jj in Ap[i]..Ap[i + 1] {
            *Bj_row = Aj[jj];
            *Bx_row = Ax[jj];
            Bj_row += 1;
            Bx_row += 1;
        }
    }
}

/// Compute C = A*B for CSR matrices A,B
///
///
/// Input Arguments:
///   I  n_row       - number of rows in A
///   I  n_col       - number of columns in B (hence C is n_row by n_col)
///   I  Ap[n_row+1] - row pointer
///   I  Aj[nnz(A)]  - column indices
///   T  Ax[nnz(A)]  - nonzeros
///   I  Bp[?]       - row pointer
///   I  Bj[nnz(B)]  - column indices
///   T  Bx[nnz(B)]  - nonzeros
/// Output Arguments:
///   I  Cp[n_row+1] - row pointer
///   I  Cj[nnz(C)]  - column indices
///   T  Cx[nnz(C)]  - nonzeros
///
/// Note:
///   Output arrays Cp, Cj, and Cx must be preallocated.
///   In order to find the appropriate type for T, csr_matmat_maxnnz can be used
///   to find nnz(C).
///
/// Note:
///   Input:  A and B column indices *are not* assumed to be in sorted order
///   Output: C column indices *are not* assumed to be in sorted order
///           Cx will not contain any zero entries
///
///   Complexity: O(n_row*K^2 + max(n_row,n_col))
///                 where K is the maximum nnz in a row of A
///                 and column of B.
///
///
///  This is an implementation of the SMMP algorithm:
///
///    "Sparse Matrix Multiplication Package (SMMP)"
///      Randolph E. Bank and Craig C. Douglas
///
///    http://citeseer.ist.psu.edu/445062.html
///    http://www.mgnet.org/~douglas/ccd-codes.html
///

/// Compute the number of non-zeroes (nnz) in the result of C = A * B.
pub fn csr_matmat_maxnnz<I: PrimInt>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Bp: &[I],
    Bj: &[I],
) -> usize /*npy_intp*/
{
    // method that uses O(n) temp storage
    // std::vector<I> mask(n_col, -1);
    let mask: Vec<I> = vec![n_col, -1];

    let /*npy_intp*/ mut nnz = 0;
    for i in 0..n_row {
        let /*npy_intp*/ mut row_nnz = 0;

        for jj in Ap[i]..Ap[i + 1] {
            let j: I = Aj[jj];
            for kk in Bp[j]..Bp[j + 1] {
                let k: I = Bj[kk];
                if mask[k] != i {
                    mask[k] = i;
                    row_nnz += 1;
                }
            }
        }

        let /*npy_intp*/ next_nnz = nnz + row_nnz;

        if row_nnz > usize::MAX - nnz {
            /*
             * Index overflowed. Note that row_nnz <= n_col and cannot overflow
             */
            panic!("nnz of the result is too large");
        }

        nnz = next_nnz;
    }

    nnz
}

/// Compute CSR entries for matrix C = A*B.
pub fn csr_matmat<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bp: &[I],
    Bj: &[I],
    Bx: &[T],
    Cp: &mut [I],
    Cj: &mut [I],
    Cx: &mut [T],
) {
    // std::vector<I> next(n_col,-1);
    let next: Vec<I> = vec![-I::one(); n_col];
    // std::vector<T> sums(n_col, 0);
    let sums: Vec<T> = vec![T::zero(); n_col];

    let mut nnz: I = I::zero();

    Cp[0] = 0;

    for i in 0..n_row {
        let mut head: I = -I::one() - I::one();
        let mut length: I = I::zero();

        let jj_start: I = Ap[i];
        let jj_end: I = Ap[i + 1];
        for jj in jj_start..jj_end {
            let j: I = Aj[jj];
            let v: T = Ax[jj];

            let kk_start: I = Bp[j];
            let kk_end: I = Bp[j + 1];
            for kk in kk_start..kk_end {
                let k: I = Bj[kk];

                sums[k] += v * Bx[kk];

                if next[k] == -1 {
                    next[k] = head;
                    head = k;
                    length += 1;
                }
            }
        }

        for _jj in 0..length {
            if sums[head] != 0 {
                Cj[nnz] = head;
                Cx[nnz] = sums[head];
                nnz += 1;
            }

            let temp: I = head;
            head = next[head];

            next[temp] = -1; //clear arrays
            sums[temp] = 0;
        }

        Cp[i + 1] = nnz;
    }
}

/// Compute C = A (binary_op) B for CSR matrices that are not
/// necessarily canonical CSR format.  Specifically, this method
/// works even when the input matrices have duplicate and/or
/// unsorted column indices within a given row.
///
/// Refer to csr_binop_csr() for additional information
///
/// Note:
///   Output arrays Cp, Cj, and Cx must be preallocated
///   If nnz(C) is not known a priori, a conservative bound is:
///          nnz(C) <= nnz(A) + nnz(B)
///
/// Note:
///   Input:  A and B column indices are not assumed to be in sorted order
///   Output: C column indices are not generally in sorted order
///           C will not contain any duplicate entries or explicit zeros.
// template <class I, class T, class T2, class binary_op>
// pub fn csr_binop_csr_general<I: PrimInt, T: Scalar, T2: Scalar, B: BinaryOp<T>>(
pub fn csr_binop_csr_general<I: PrimInt, T: Scalar, T2: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bp: &[I],
    Bj: &[I],
    Bx: &[T],
    Cp: &mut [I],
    Cj: &mut [I],
    Cx: &mut [T2],
    op: Binop<T>,
) {
    // Method that works for duplicate and/or unsorted indices

    // std::vector<I>  next(n_col,-1);
    let mut next: Vec<I> = vec![-I::one(); n_col];
    // std::vector<T> A_row(n_col, 0);
    let mut A_row: Vec<T> = vec![T::zero(); n_col];
    // std::vector<T> B_row(n_col, 0);
    let mut B_row: Vec<T> = vec![T::zero(); n_col];

    let mut nnz: I = I::zero();
    Cp[0] = 0;

    for i in 0..n_row {
        let mut head: I = -I::one() - I::one();
        let mut length: I = I::zero();

        //add a row of A to A_row
        let i_start: I = Ap[i];
        let i_end: I = Ap[i + 1];
        for jj in i_start..i_end {
            let j: I = Aj[jj];

            A_row[j] += Ax[jj];

            if next[j] == -1 {
                next[j] = head;
                head = j;
                length += 1;
            }
        }

        //add a row of B to B_row
        let i_start = Bp[i];
        let i_end = Bp[i + 1];
        for jj in i_start..i_end {
            let j: I = Bj[jj];

            B_row[j] += Bx[jj];

            if next[j] == -1 {
                next[j] = head;
                head = j;
                length += 1;
            }
        }

        // scan through columns where A or B has
        // contributed a non-zero entry
        for _jj in 0..length {
            let result: T = op.binary_op(A_row[head], B_row[head]);

            if result != 0 {
                Cj[nnz] = head;
                Cx[nnz] = result;
                nnz += 1;
            }

            let temp: I = head;
            head = next[head];

            next[temp] = -I::one();
            A_row[temp] = 0;
            B_row[temp] = 0;
        }

        Cp[i + 1] = nnz;
    }
}

/// Compute C = A (binary_op) B for CSR matrices that are in the
/// canonical CSR format.  Specifically, this method requires that
/// the rows of the input matrices are free of duplicate column indices
/// and that the column indices are in sorted order.
///
/// Refer to csr_binop_csr() for additional information
///
/// Note:
///   Input:  A and B column indices are assumed to be in sorted order
///   Output: C column indices will be in sorted order
///           Cx will not contain any zero entries
// template <class I, class T, class T2, class binary_op>
// pub fn csr_binop_csr_canonical<I: PrimInt, T: Scalar, T2: Scalar, B: BinaryOp<T>>(
pub fn csr_binop_csr_canonical<I: PrimInt, T: Scalar, T2: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bp: &[I],
    Bj: &[I],
    Bx: &[T],
    Cp: &mut [I],
    Cj: &mut [I],
    Cx: &mut [T2],
    op: Binop<T>,
) {
    //Method that works for canonical CSR matrices

    Cp[0] = 0;
    let mut nnz: I = I::zero();

    for i in 0..n_row {
        let mut A_pos: I = Ap[i];
        let mut B_pos: I = Bp[i];
        let A_end: I = Ap[i + 1];
        let B_end: I = Bp[i + 1];

        //while not finished with either row
        while A_pos < A_end && B_pos < B_end {
            let A_j: I = Aj[A_pos];
            let B_j: I = Bj[B_pos];

            if A_j == B_j {
                let result: T = op.binary_op(Ax[A_pos], Bx[B_pos]);
                if result != 0 {
                    Cj[nnz] = A_j;
                    Cx[nnz] = result;
                    nnz += 1;
                }
                A_pos += 1;
                B_pos += 1;
            } else if A_j < B_j {
                let result: T = op.binary_op(Ax[A_pos], 0);
                if result != 0 {
                    Cj[nnz] = A_j;
                    Cx[nnz] = result;
                    nnz += 1;
                }
                A_pos += 1;
            } else {
                //B_j < A_j
                let result: T = op.binary_op(0, Bx[B_pos]);
                if result != 0 {
                    Cj[nnz] = B_j;
                    Cx[nnz] = result;
                    nnz += 1;
                }
                B_pos += 1;
            }
        }

        //tail
        while A_pos < A_end {
            let result: T = op.binary_op(Ax[A_pos], 0);
            if result != 0 {
                Cj[nnz] = Aj[A_pos];
                Cx[nnz] = result;
                nnz += 1;
            }
            A_pos += 1;
        }
        while B_pos < B_end {
            let result: T = op.binary_op(0, Bx[B_pos]);
            if result != 0 {
                Cj[nnz] = Bj[B_pos];
                Cx[nnz] = result;
                nnz += 1;
            }
            B_pos += 1;
        }

        Cp[i + 1] = nnz;
    }
}

/// Compute C = A (binary_op) B for CSR matrices A,B where the column
/// indices with the rows of A and B are known to be sorted.
///
///   binary_op(x,y) - binary operator to apply elementwise
///
/// Input Arguments:
///   I    n_row       - number of rows in A (and B)
///   I    n_col       - number of columns in A (and B)
///   I    Ap[n_row+1] - row pointer
///   I    Aj[nnz(A)]  - column indices
///   T    Ax[nnz(A)]  - nonzeros
///   I    Bp[n_row+1] - row pointer
///   I    Bj[nnz(B)]  - column indices
///   T    Bx[nnz(B)]  - nonzeros
/// Output Arguments:
///   I    Cp[n_row+1] - row pointer
///   I    Cj[nnz(C)]  - column indices
///   T    Cx[nnz(C)]  - nonzeros
///
/// Note:
///   Output arrays Cp, Cj, and Cx must be preallocated
///   If nnz(C) is not known a priori, a conservative bound is:
///          nnz(C) <= nnz(A) + nnz(B)
///
/// Note:
///   Input:  A and B column indices are not assumed to be in sorted order.
///   Output: C column indices will be in sorted if both A and B have sorted indices.
///           Cx will not contain any zero entries
// template <class I, class T, class T2, class binary_op>
// pub fn csr_binop_csr<I: PrimInt, T: Scalar, T2: Scalar, B: BinaryOp<T>>(
pub fn csr_binop_csr<I: PrimInt, T: Scalar, T2: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bp: &[I],
    Bj: &[I],
    Bx: &[T],
    Cp: &mut [I],
    Cj: &mut [I],
    Cx: &mut [T2],
    op: Binop<T>,
) {
    if csr_has_canonical_format(n_row, Ap, Aj) && csr_has_canonical_format(n_row, Bp, Bj) {
        csr_binop_csr_canonical(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, op);
    } else {
        // Scalar::ne()
        csr_binop_csr_general(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, op);
    }
}

// element-wise binary operations //

// template <class I, class T, class T2>
pub fn csr_ne_csr<I: PrimInt, T: Scalar, T2: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bp: &[I],
    Bj: &[I],
    Bx: &[T],
    Cp: &mut [I],
    Cj: &mut [I],
    Cx: &mut [T2],
) {
    // csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, Scalar::ne);
    csr_binop_csr(
        n_row,
        n_col,
        Ap,
        Aj,
        Ax,
        Bp,
        Bj,
        Bx,
        Cp,
        Cj,
        Cx,
        Scalar::add,
    );
}

/// Sum together duplicate column entries in each row of CSR matrix A
///
///
/// Input Arguments:
///   I    n_row       - number of rows in A (and B)
///   I    n_col       - number of columns in A (and B)
///   I    Ap[n_row+1] - row pointer
///   I    Aj[nnz(A)]  - column indices
///   T    Ax[nnz(A)]  - nonzeros
///
/// Note:
///   The column indices within each row must be in sorted order.
///   Explicit zeros are retained.
///   Ap, Aj, and Ax will be modified *inplace*
pub fn csr_sum_duplicates<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &mut [I],
    Aj: &mut [I],
    Ax: &mut [T],
) {
    let mut nnz: I = 0;
    let mut row_end: I = 0;
    for i in 0..n_row {
        let mut jj: I = row_end;
        row_end = Ap[i + 1];
        while jj < row_end {
            let j: I = Aj[jj];
            let mut x: T = Ax[jj];
            jj += 1;
            while jj < row_end && Aj[jj] == j {
                x += Ax[jj];
                jj += 1;
            }
            Aj[nnz] = j;
            Ax[nnz] = x;
            nnz += 1;
        }
        Ap[i + 1] = nnz;
    }
}

/// Eliminate zero entries from CSR matrix A
///
///
/// Input Arguments:
///   I    n_row       - number of rows in A (and B)
///   I    n_col       - number of columns in A (and B)
///   I    Ap[n_row+1] - row pointer
///   I    Aj[nnz(A)]  - column indices
///   T    Ax[nnz(A)]  - nonzeros
///
/// Note:
///   Ap, Aj, and Ax will be modified *inplace*
pub fn csr_eliminate_zeros<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &mut [I],
    Aj: &mut [I],
    Ax: &mut [T],
) {
    let mut nnz: I = 0;
    let mut row_end: I = 0;
    for i in 0..n_row {
        let mut jj: I = row_end;
        row_end = Ap[i + 1];
        while jj < row_end {
            let j: I = Aj[jj];
            let x: T = Ax[jj];
            if x != 0 {
                Aj[nnz] = j;
                Ax[nnz] = x;
                nnz += 1;
            }
            jj += 1;
        }
        Ap[i + 1] = nnz;
    }
}

/// Compute Y += A*X for CSR matrix A and dense vectors X,Y
///
///
/// Input Arguments:
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   T  Ax[nnz(A)]    - nonzeros
///   T  Xx[n_col]     - input vector
///
/// Output Arguments:
///   T  Yx[n_row]     - output vector
///
/// Note:
///   Output array Yx must be preallocated
///
///   Complexity: Linear.  Specifically O(nnz(A) + n_row)
pub fn csr_matvec<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Xx: &[T],
    Yx: &mut [T],
) {
    for i in 0..n_row {
        let mut sum: T = Yx[i];
        for jj in Ap[i]..Ap[i + 1] {
            sum += Ax[jj] * Xx[Aj[jj]];
        }
        Yx[i] = sum;
    }
}

/// Compute Y += A*X for CSR matrix A and dense block vectors X,Y
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
pub fn csr_matvecs<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    n_vecs: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Xx: &[T],
    Yx: &mut [T],
) {
    for i in 0..n_row {
        let y = Yx + /*(npy_intp)*/n_vecs * i;
        for jj in Ap[i]..Ap[i + 1] {
            let j: I = Aj[jj];
            let a: T = Ax[jj];
            let x = Xx + /*(npy_intp)*/n_vecs * j;
            axpy(n_vecs, a, x, y);
        }
    }
}

pub fn csr_select<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    rowidx: &[I],
    colidx: &[I],
    Bp: &mut Vec<I>,
    Bj: &mut Vec<I>,
    Bx: &mut Vec<T>,
) {
    let new_n_row: I = rowidx.len();
    let mut new_nnz: I = 0;

    // Count nonzeros total/per row.
    for row in rowidx {
        let row_start: I = Ap[row];
        let row_end: I = Ap[row + 1];

        let mut colptrs = HashMap::new(); // TODO: Cache for second pass?

        for jj in row_start..row_end {
            colptrs.insert(Aj[jj], jj);
        }
        for col in colidx {
            if colptrs.has_entry(col) {
                new_nnz += 1;
            }
        }
    }

    // Allocate.
    Bp.resize(new_n_row + 1, I::zero());
    Bj.resize(new_nnz, I::zero());
    Bx.resize(new_nnz, T::zero());

    // Assign.
    Bp[0] = 0;
    let mut kk = 0;
    for (r, row) in rowidx.iter().enumerate() {
        let start = Ap[row];
        let end = Ap[row + 1];

        let mut colptrs = HashMap::new();
        for jj in start..end {
            colptrs.insert(Aj[jj], jj);
        }

        for (c, col) in colidx.iter().enumerate() {
            if let Some(jj) = colptrs.get(col) {
                Bj[kk] = c;
                Bx[kk] = Ax[jj];
                kk += 1;
            }
        }
        Bp[r + 1] = kk;
    }
}

pub fn get_csr_submatrix<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    ir0: I,
    ir1: I,
    ic0: I,
    ic1: I,
    Bp: &mut Vec<I>,
    Bj: &mut Vec<I>,
    Bx: &mut Vec<T>,
) {
    let new_n_row: I = ir1 - ir0;
    //I new_n_col = ic1 - ic0;  //currently unused
    let mut new_nnz: I = 0;
    let mut kk: I = 0;

    // Count nonzeros total/per row.
    for i in 0..new_n_row {
        let row_start: I = Ap[ir0 + i];
        let row_end: I = Ap[ir0 + i + 1];

        for jj in row_start..row_end {
            if (Aj[jj] >= ic0) && (Aj[jj] < ic1) {
                new_nnz += 1;
            }
        }
    }

    // Allocate.
    Bp.resize(new_n_row + 1, I::zero());
    Bj.resize(new_nnz, I::zero());
    Bx.resize(new_nnz, T::zero());

    // Assign.
    // (*Bp)[0] = 0;
    Bp[0] = 0;
    for i in 0..new_n_row {
        let row_start: I = Ap[ir0 + i];
        let row_end: I = Ap[ir0 + i + 1];

        for jj in row_start..row_end {
            if (Aj[jj] >= ic0) && (Aj[jj] < ic1) {
                // (*Bj)[kk] = Aj[jj] - ic0;
                Bj[kk] = Aj[jj] - ic0;
                // (*Bx)[kk] = Ax[jj];
                Bx[kk] = Ax[jj];
                kk += 1;
            }
        }
        // (*Bp)[i+1] = kk;
        Bp[i + 1] = kk;
    }
}

/// Slice rows given as an array of indices.
///
/// Input Arguments:
///   I  n_row_idx       - number of row indices
///   I  rows[n_row_idx] - row indices for indexing
///   I  Ap[n_row+1]     - row pointer
///   I  Aj[nnz(A)]      - column indices
///   T  Ax[nnz(A)]      - data
///
/// Output Arguments:
///   I  Bj - new column indices
///   T  Bx - new data
pub fn csr_row_index<I: PrimInt, T: Scalar>(
    n_row_idx: I,
    rows: &[I],
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bj: &mut Vec<I>,
    Bx: &mut Vec<T>,
) {
    for i in 0..n_row_idx {
        let row: I = rows[i];
        let row_start: I = Ap[row];
        let row_end: I = Ap[row + 1];
        // Bj = copy(Aj + row_start, Aj + row_end, Bj);
        Bj.extend_from_slice(Aj[row_start..row_end]);
        // Bx = copy(Ax + row_start, Ax + row_end, Bx);
        Bx.extend_from_slice(Ax[row_start..row_end]);
    }
}

/// Slice rows given as a (start, stop, step) tuple.
///
/// Input Arguments:
///   I  start
///   I  stop
///   I  step
///   I  Ap[N+1]    - row pointer
///   I  Aj[nnz(A)] - column indices
///   T  Ax[nnz(A)] - data
///
/// Output Arguments:
///   I  Bj - new column indices
///   T  Bx - new data
pub fn csr_row_slice<I: PrimInt, T: Scalar>(
    start: I,
    stop: I,
    step: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bj: &mut Vec<I>,
    Bx: &mut Vec<T>,
) {
    if step > 0 {
        let mut row = start;
        while row < stop {
            let row_start: I = Ap[row];
            let row_end: I = Ap[row + 1];
            // Bj = copy(Aj + row_start, Aj + row_end, Bj);
            Bj.extend_from_slice(Aj[row_start..row_end]);
            // Bx = copy(Ax + row_start, Ax + row_end, Bx);
            Bx.extend_from_slice(Ax[row_start..row_end]);
            row += step;
        }
    } else {
        let mut row: I = start;
        while row > stop {
            let row_start: I = Ap[row];
            let row_end: I = Ap[row + 1];
            // Bj = copy(Aj + row_start, Aj + row_end, Bj);
            Bj.extend_from_slice(Aj[row_start..row_end]);
            // Bx = copy(Ax + row_start, Ax + row_end, Bx);
            Bx.extend_from_slice(Ax[row_start..row_end]);
            row += step;
        }
    }
}

/// Slice columns given as an array of indices (pass 1).
/// This pass counts idx entries and computes a new indptr.
///
/// Input Arguments:
///   I  n_idx           - number of indices to slice
///   I  col_idxs[n_idx] - indices to slice
///   I  n_row           - major axis dimension
///   I  n_col           - minor axis dimension
///   I  Ap[n_row+1]     - indptr
///   I  Aj[nnz(A)]      - indices
///
/// Output Arguments:
///   I  col_offsets[n_col] - cumsum of index repeats
///   I  Bp[n_row+1]        - new indptr
pub fn csr_column_index1<I: PrimInt>(
    n_idx: I,
    col_idxs: &[I],
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    col_offsets: &mut [I],
    Bp: &mut [I],
) {
    // bincount(col_idxs)
    for jj in 0..n_idx {
        let j: I = col_idxs[jj];
        col_offsets[j] += 1;
    }

    // Compute new indptr
    let mut new_nnz: I = 0;
    Bp[0] = 0;
    for i in 0..n_row {
        for jj in Ap[i]..Ap[i + 1] {
            new_nnz += col_offsets[Aj[jj]];
        }
        Bp[i + 1] = new_nnz;
    }

    // cumsum in-place
    for j in 1..n_col {
        col_offsets[j] += col_offsets[j - 1];
    }
}

/// Slice columns given as an array of indices (pass 2).
/// This pass populates indices/data entries for selected columns.
///
/// Input Arguments:
///   I  col_order[n_idx]   - order of col indices
///   I  col_offsets[n_col] - cumsum of col index counts
///   I  nnz                - nnz(A)
///   I  Aj[nnz(A)]         - column indices
///   T  Ax[nnz(A)]         - data
///
/// Output Arguments:
///   I  Bj[nnz(B)] - new column indices
///   T  Bx[nnz(B)] - new data
pub fn csr_column_index2<I: PrimInt, T: Scalar>(
    col_order: &[I],
    col_offsets: &[I],
    nnz: I,
    Aj: &[I],
    Ax: &[T],
    Bj: &mut [I],
    Bx: &mut [T],
) {
    let mut n: I = 0;
    for jj in 0..nnz {
        let j: I = Aj[jj];
        let offset: I = col_offsets[j];
        let prev_offset: I = if j == 0 { 0 } else { col_offsets[j - 1] };
        if offset != prev_offset {
            let v: T = Ax[jj];
            for k in prev_offset..offset {
                Bj[n] = col_order[k];
                Bx[n] = v;
                n += 1;
            }
        }
    }
}

/// Count the number of occupied diagonals in CSR matrix A
///
/// Input Arguments:
///   I  nnz             - number of nonzeros in A
///   I  Ai[nnz(A)]      - row indices
///   I  Aj[nnz(A)]      - column indices
pub fn csr_count_diagonals<I: PrimInt>(n_row: I, Ap: &[I], Aj: &[I]) {
    let mut diagonals = HashSet::<I>::new();

    for i in 0..n_row {
        for jj in Ap[i]..Ap[i + 1] {
            diagonals.insert(Aj[jj] - i);
        }
    }
    return diagonals.size();
}

/// Sample the matrix at specific locations
///
/// Determine the matrix value for each row,col pair
///    Bx[n] = A(Bi[n],Bj[n])
///
/// Input Arguments:
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   T  Ax[nnz(A)]    - nonzeros
///   I  n_samples     - number of samples
///   I  Bi[N]         - sample rows
///   I  Bj[N]         - sample columns
///
/// Output Arguments:
///   T  Bx[N]         - sample values
///
/// Note:
///   Output array Bx must be preallocated
///
///   Complexity: varies
///
///   TODO handle other cases with asymptotically optimal method
pub fn csr_sample_values<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    n_samples: I,
    Bi: &[I],
    Bj: &[I],
    Bx: &mut [T],
) {
    // ideally we'd do the following
    // Case 1: A is canonical and B is sorted by row and column
    //   -> special purpose csr_binop_csr() (optimized form)
    // Case 2: A is canonical and B is unsorted and max(log(Ap[i+1] - Ap[i])) > log(num_samples)
    //   -> do binary searches for each sample
    // Case 3: A is canonical and B is unsorted and max(log(Ap[i+1] - Ap[i])) < log(num_samples)
    //   -> sort B by row and column and use Case 1
    // Case 4: A is not canonical and num_samples ~ nnz
    //   -> special purpose csr_binop_csr() (general form)
    // Case 5: A is not canonical and num_samples << nnz
    //   -> do linear searches for each sample

    let nnz: I = Ap[n_row];

    let threshold: I = nnz / 10; // constant is arbitrary

    if n_samples > threshold && csr_has_canonical_format(n_row, Ap, Aj) {
        for n in 0..n_samples {
            let i: I = if Bi[n] < 0 { Bi[n] + n_row } else { Bi[n] }; // sample row
            let j: I = if Bj[n] < 0 { Bj[n] + n_col } else { Bj[n] }; // sample column

            let row_start: I = Ap[i];
            let row_end: I = Ap[i + 1];

            if row_start < row_end {
                // let offset: I = lower_bound(Aj + row_start, Aj + row_end, j) - Aj;
                let offset: I = Aj[row_start..row_end].lower_bound(j) - Aj;

                if offset < row_end && Aj[offset] == j {
                    Bx[n] = Ax[offset];
                } else {
                    Bx[n] = 0;
                }
            } else {
                Bx[n] = 0;
            }
        }
    } else {
        for n in 0..n_samples {
            let i: I = if Bi[n] < 0 { Bi[n] + n_row } else { Bi[n] }; // sample row
            let j: I = if Bj[n] < 0 { Bj[n] + n_col } else { Bj[n] }; // sample column

            let row_start: I = Ap[i];
            let row_end: I = Ap[i + 1];

            let mut x: T = 0;

            for jj in row_start..row_end {
                if Aj[jj] == j {
                    x += Ax[jj];
                }
            }

            Bx[n] = x;
        }
    }
}

/// Determine the data offset at specific locations
///
/// Input Arguments:
///   I  n_row         - number of rows in A
///   I  n_col         - number of columns in A
///   I  Ap[n_row+1]   - row pointer
///   I  Aj[nnz(A)]    - column indices
///   I  n_samples     - number of samples
///   I  Bi[N]         - sample rows
///   I  Bj[N]         - sample columns
///
/// Output Arguments:
///   I  Bp[N]         - offsets into Aj; -1 if non-existent
///
/// Return value:
///   1 if any sought entries are duplicated, in which case the
///   function has exited early; 0 otherwise.
///
/// Note:
///   Output array Bp must be preallocated
///
///   Complexity: varies. See csr_sample_values
pub fn csr_sample_offsets<I: PrimInt>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    n_samples: I,
    Bi: &[I],
    Bj: &[I],
    Bp: &mut [I],
) -> usize {
    let mut nnz: I = Ap[n_row];
    let threshold: I = nnz / 10; // constant is arbitrary

    if n_samples > threshold && csr_has_canonical_format(n_row, Ap, Aj) {
        for n in 0..n_samples {
            let i: I = if Bi[n] < 0 { Bi[n] + n_row } else { Bi[n] }; // sample row
            let j: I = if Bj[n] < 0 { Bj[n] + n_col } else { Bj[n] }; // sample column

            let row_start: I = Ap[i];
            let row_end: I = Ap[i + 1];

            if row_start < row_end {
                // let offset: I = lower_bound(Aj + row_start, Aj + row_end, j) - Aj;
                let offset: I = Aj[row_start..row_end].lower_bound(j) - Aj;

                if offset < row_end && Aj[offset] == j {
                    Bp[n] = offset;
                } else {
                    Bp[n] = -1;
                }
            } else {
                Bp[n] = -1;
            }
        }
    } else {
        for n in 0..n_samples {
            let i: I = if Bi[n] < 0 { Bi[n] + n_row } else { Bi[n] }; // sample row
            let j: I = if Bj[n] < 0 { Bj[n] + n_col } else { Bj[n] }; // sample column

            let row_start: I = Ap[i];
            let row_end: I = Ap[i + 1];

            let mut offset: I = -1;

            let mut jj = row_start; // FIXME: check
            while jj < row_end {
                if Aj[jj] == j {
                    offset = jj;
                    jj += 1;
                    while jj < row_end {
                        if Aj[jj] == j {
                            offset = -2;
                            return 1;
                        }
                        jj += 1;
                    }
                }
                jj += 1;
            }
            Bp[n] = offset;
        }
    }
    return 0;
}

/// Stack CSR matrices in A horizontally (column wise)
///
/// Input Arguments:
///   I  n_blocks                      - number of matrices in A
///   I  n_row                         - number of rows in any matrix in A
///   I  n_col_cat[n_blocks]           - number of columns in each matrix in A concatenated
///   I  Ap_cat[n_blocks*(n_row + 1)]  - row indices of each matrix in A concatenated
///   I  Aj_cat[nnz(A)]                - column indices of each matrix in A concatenated
///   T  Ax_cat[nnz(A)]                - nonzeros of each matrix in A concatenated
///
/// Output Arguments:
///   I Bp  - row pointer
///   I Bj  - column indices
///   T Bx  - nonzeros
///
/// Note:
///   All output arrays Bp, Bj, Bx must be preallocated
///
///   Complexity: Linear.  Specifically O(nnz(A) + n_blocks)
pub fn csr_hstack<I: PrimInt, T: Scalar>(
    n_blocks: I,
    n_row: I,
    n_col_cat: &[I],
    Ap_cat: &[I],
    Aj_cat: &[I],
    Ax_cat: &[T],
    Bp: &mut [I],
    Bj: &mut [I],
    Bx: &mut [T],
) {
    // First, mark the blocks in the input data while
    // computing their column offsets:
    let mut col_offset: Vec<I> = vec![I::zero(); n_blocks];
    // std::vector<const I*> bAp(n_blocks);
    let mut bAp = vec![n_blocks];
    // std::vector<const I*> bAj(n_blocks);
    let mut bAj = vec![n_blocks];
    // std::vector<const T*> bAx(n_blocks);
    let mut bAx = vec![n_blocks];
    col_offset[0] = 0;
    bAp[0] = Ap_cat;
    bAj[0] = Aj_cat;
    bAx[0] = Ax_cat;
    for b in 1..n_blocks {
        col_offset[b] = col_offset[b - 1] + n_col_cat[b - 1];
        bAp[b] = bAp[b - 1] + (n_row + 1);
        bAj[b] = bAj[b - 1] + bAp[b - 1][n_row];
        bAx[b] = bAx[b - 1] + bAp[b - 1][n_row];
    }

    // Next, build the full output matrix:
    Bp[0] = 0;
    let mut s: I = 0;
    for i in 0..n_row {
        for b in 0..n_blocks {
            let jj_start: I = bAp[b][i];
            let jj_end: I = bAp[b][i + 1];
            let offset: I = col_offset[b];
            // FIXME
            // transform(&bAj[b][jj_start], &bAj[b][jj_end], &Bj[s], [&](I x){return (x + offset);});
            // copy(&bAx[b][jj_start], &bAx[b][jj_end], &Bx[s]);
            s += jj_end - jj_start;
        }
        Bp[i + 1] = s;
    }
}

pub fn csr_tocoo<I: PrimInt, T: Scalar>(
    n_row: I,
    n_col: I,
    Ap: &[I],
    Aj: &[I],
    Ax: &[T],
    Bi: &mut [I],
    Bj: &mut [I],
    Bx: &mut [T],
) {
    let mut k = 0;
    for i in 0..n_row {
        let (start, end) = (Ap[i], Ap[i + 1]);
        for jj in start..end {
            Bi[k] = i;
            Bj[k] = Aj[jj];
            Bx[k] = Ax[jj];
            k += 1;
        }
    }
}
