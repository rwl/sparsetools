use crate::csr::CSR;
use crate::traits::{Complex, Float, Integer};

pub trait CCSR<I: Integer, F: Float> {
    fn conj(&self) -> Self;
    fn real(&self) -> CSR<I, F>;
    fn imag(&self) -> CSR<I, F>;
}

impl<I, F, C> CCSR<I, F> for CSR<I, C>
where
    I: Integer,
    F: Float,
    C: Complex<F>,
{
    fn conj(&self) -> CSR<I, C> {
        CSR {
            rows: self.rows,
            cols: self.cols,
            rowptr: self.rowptr.clone(),
            colidx: self.colidx.clone(),
            data: self.data.iter().map(|d| d.conj()).collect(),
        }
    }

    fn real(&self) -> CSR<I, F> {
        CSR {
            rows: self.rows,
            cols: self.cols,
            rowptr: self.rowptr.clone(),
            colidx: self.colidx.clone(),
            data: self.data.iter().map(|d| d.real()).collect(),
        }
    }

    fn imag(&self) -> CSR<I, F> {
        CSR {
            rows: self.rows,
            cols: self.cols,
            rowptr: self.rowptr.clone(),
            colidx: self.colidx.clone(),
            data: self.data.iter().map(|d| d.imag()).collect(),
        }
    }
}
