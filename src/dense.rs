use crate::scalar::Scalar;
use crate::util::Binop;
use num_traits::PrimInt;

// Simplified BLAS routines and other dense linear algebra functions

// Level 1 //

/// y += a*x
pub(crate) fn axpy<I: PrimInt, T: Scalar>(n: I, a: T, x: &[T], y: &mut [T]) {
    for i in 0..n {
        y[i] += a * x[i];
    }
}

/// scale a vector in-place
pub(crate) fn scal<I: PrimInt, T: Scalar>(n: I, a: T, x: &mut [T]) {
    for i in 0..n {
        x[i] *= a;
    }
}

/// dot product
pub(crate) fn dot<I: PrimInt, T: Scalar>(n: I, x: &[T], y: &[T]) {
    let mut dp: T = T::zero();
    for i in 0..n {
        dp += x[i] * y[i];
    }
    dp
}

/// vectorize a binary operation
pub(crate) fn vector_binop<I: PrimInt, T: Scalar>(
    n: I,
    x: &[T],
    y: &[T],
    z: &mut [T],
    op: Binop<T>,
) {
    for i in 0..n {
        z[i] = op(x[i], y[i]);
    }
}

// Level 2 //

pub(crate) fn gemv<I: PrimInt, T: Scalar>(m: I, n: I, A: &[T], x: &[T], y: &mut [T]) {
    for i in 0..m {
        let mut dot: T = y[i];
        for j in 0..n {
            dot += A[/*(npy_intp)*/n * i + j] * x[j];
        }
        y[i] = dot;
    }
}

// Level 3 //

pub(crate) fn gemm<I: PrimInt, T: Scalar>(m: I, n: I, k: I, A: &[T], B: &[T], C: &mut [T]) {
    for i in 0..m {
        for j in 0..n {
            let mut dot: T = C[/*(npy_intp)*/n * i + j];
            for _d in 0..k {
                dot += A[/*(npy_intp)*/k * i + _d] * B[/*(npy_intp)*/n * _d + j];
            }
            C[/*(npy_intp)*/n * i + j] = dot;
        }
    }
}
