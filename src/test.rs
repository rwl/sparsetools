use std::{fmt::Display, ops::Sub};

use anyhow::{format_err, Result};
use num_complex::Complex;

// [11,  0,  0,  0]
// [21, 22,  0,  0]
// [ 0, 32, 33,  0]
// [41,  0, 43, 44]

pub(crate) const N: usize = 4;
pub(crate) const NNZ: usize = 8;

pub(crate) fn diagonal() -> Vec<f64> {
    vec![11.0, 22.0, 33.0, 44.0]
}

pub(crate) fn c_diagonal() -> Vec<Complex<f64>> {
    to_complex(&diagonal())
}

pub(crate) fn dense_data() -> Vec<Vec<f64>> {
    vec![
        vec![11.0, 0.0, 0.0, 0.0],
        vec![21.0, 22.0, 0.0, 0.0],
        vec![0.0, 32.0, 33.0, 0.0],
        vec![41.0, 0.0, 43.0, 44.0],
    ]
}
/*
pub(crate) fn c_dense_data() -> Vec<Vec<Complex<f64>>> {
    let d = dense_data();
    let mut c = Vec::with_capacity(d.len());
    for row in &d {
        c.push(to_complex(row));
    }
    c
}
*/
pub(crate) fn coo_data() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let rowidx = vec![0, 1, 3, 1, 2, 2, 3, 3];
    let colidx = vec![0, 0, 0, 1, 1, 2, 2, 3];
    let values = vec![11.0, 21.0, 41.0, 22.0, 32.0, 33.0, 43.0, 44.0];
    (rowidx, colidx, values)
}

pub(crate) fn csr_data() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let rowptr = vec![0, 1, 3, 5, 8];
    let colidx = vec![0, 0, 1, 1, 2, 0, 2, 3];
    let values = vec![11.0, 21.0, 22.0, 32.0, 33.0, 41.0, 43.0, 44.0];
    (rowptr, colidx, values)
}

pub(crate) fn csc_data() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let rowidx = vec![0, 1, 3, 1, 2, 2, 3, 3];
    let colptr = vec![0, 3, 5, 7, 8];
    let values = vec![11.0, 21.0, 41.0, 22.0, 32.0, 33.0, 43.0, 44.0];
    (rowidx, colptr, values)
}
/*
pub(crate) fn c_coo_data() -> (Vec<usize>, Vec<usize>, Vec<Complex<f64>>) {
    let (rowidx, colidx, values) = coo_data();
    (rowidx, colidx, to_complex(&values))
}
*/
pub(crate) fn c_csr_data() -> (Vec<usize>, Vec<usize>, Vec<Complex<f64>>) {
    let (rowptr, colidx, values) = csr_data();
    (rowptr, colidx, to_complex(&values))
}
/*
pub(crate) fn c_csc_data() -> (Vec<usize>, Vec<usize>, Vec<Complex<f64>>) {
    let (rowidx, colptr, values) = csc_data();
    (rowidx, colptr, to_complex(&values))
}
*/
fn to_complex(floats: &[f64]) -> Vec<Complex<f64>> {
    floats.iter().map(|&f| Complex::new(f, 0.0)).collect()
}

pub(crate) fn assert_slice<S>(actual: &[S], expected: &[S]) -> Result<()>
where
    S: Copy + PartialEq + Sub<Output = S> + Display,
{
    if actual.len() != expected.len() {
        return Err(format_err!(
            "slice length, expected = {} actual = {}, diff = {}",
            expected.len(),
            actual.len(),
            expected.len() - actual.len()
        ));
    }
    for (i, (&act, &exp)) in actual.iter().zip(expected).enumerate() {
        if act != exp {
            return Err(format_err!(
                "element {}/{}, expected = {} actual = {} diff = {}",
                i,
                actual.len(),
                exp,
                act,
                exp - act
            ));
        }
    }
    Ok(())
}

pub(crate) fn assert_dense<S>(actual: &Vec<Vec<S>>, expected: &Vec<Vec<S>>) -> Result<()>
where
    S: Copy + PartialEq + Sub<Output = S> + Display,
{
    if actual.len() != expected.len() {
        return Err(format_err!(
            "dense rows, expected = {} actual = {}, diff = {}",
            expected.len(),
            actual.len(),
            expected.len() - actual.len()
        ));
    }
    for (i, (act, exp)) in actual.iter().zip(expected).enumerate() {
        assert_slice(act, exp).map_err(|err| format_err!("row {}: {}", i, err))?;
    }
    Ok(())
}
