use crate::traits::{Integer, Scalar};

// Simplified BLAS routines and other dense linear algebra functions

// Level 1 //

/// y += a*x
pub(crate) fn axpy<I: Integer, T: Scalar>(n: I, a: T, x: &[T], y: &mut [T]) {
    for i in 0..n.to_usize().unwrap() {
        y[i] += a * x[i];
    }
}
/*
/// scale a vector in-place
pub(crate) fn scal<I: Integer, T: Scalar>(n: I, a: T, x: &mut [T]) {
    for i in 0..n.to_usize().unwrap() {
        x[i] *= a;
    }
}

/// dot product
pub(crate) fn dot<I: Integer, T: Scalar>(n: I, x: &[T], y: &[T]) -> T {
    let mut dp: T = T::zero();
    for i in 0..n.to_usize().unwrap() {
        dp += x[i] * y[i];
    }
    dp
}

/// vectorize a binary operation
pub(crate) fn vector_binop<I: Integer, T: Scalar>(
    n: I,
    x: &[T],
    y: &[T],
    z: &mut [T],
    op: Binop<T, T>,
) {
    for i in 0..n.to_usize().unwrap() {
        z[i] = op(x[i], y[i]);
    }
}

// Level 2 //

pub(crate) fn gemv<I: Integer, T: Scalar>(m: I, n: I, A: &[T], x: &[T], y: &mut [T]) {
    for i in 0..m.to_usize().unwrap() {
        let mut dot: T = y[i];
        for j in 0..n.to_usize().unwrap() {
            let k = /*(npy_intp)*/n.to_usize().unwrap() * i + j;
            dot += A[k] * x[j];
        }
        y[i] = dot;
    }
}

// Level 3 //

pub(crate) fn gemm<I: Integer, T: Scalar>(m: I, n: I, k: I, A: &[T], B: &[T], C: &mut [T]) {
    let un = n.to_usize().unwrap();
    for i in 0..m.to_usize().unwrap() {
        for j in 0..n.to_usize().unwrap() {
            let mut dot: T = C[/*(npy_intp)*/un * i + j];
            for _d in 0..k.to_usize().unwrap() {
                dot += A[k.to_usize().unwrap() * i + _d] * B[un * _d + j];
            }
            C[/*(npy_intp)*/un * i + j] = dot;
        }
    }
}
*/
