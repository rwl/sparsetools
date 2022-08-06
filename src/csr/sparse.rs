use crate::coo::Coo;
use crate::csr::CSR;
use crate::traits::{Complex, Float, Integer};

pub fn sparse<I: Integer, F: Float, C: Complex<F>>(
    rows: usize,
    cols: usize,
    rowidx: &[usize],
    colidx: &[usize],
    data: &[C],
) -> CSR<I, C> {
    Coo::new(rows, cols, rowidx.to_vec(), colidx.to_vec(), data.to_vec()).to_csr()
}

pub fn spdiag<I: Integer, F: Float, C: Complex<F>>(d: &[C]) -> CSR<I, C> {
    let n = d.len();
    Coo::new(n, n, (0..n).collect(), (0..n).collect(), d.to_vec()).to_csr()
}
