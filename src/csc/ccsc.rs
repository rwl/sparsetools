use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Complex, Float, Integer};

pub trait CCSC<I, F, C> {
    fn conj(&self) -> Self;
    fn real(&self) -> CSC<I, F>;
    fn imag(&self) -> CSC<I, F>;
    /// Returns the conjugate transpose of self.
    fn h(&self) -> CSR<I, C>;
}

impl<I, F, C> CCSC<I, F, C> for CSC<I, C>
where
    I: Integer,
    F: Float,
    C: Complex<F>,
{
    fn conj(&self) -> CSC<I, C> {
        CSC::new(
            self.rows(),
            self.cols(),
            self.rowidx().to_vec(),
            self.colptr().to_vec(),
            self.values().iter().map(|d| d.conj()).collect(),
        )
        .unwrap()
    }

    fn real(&self) -> CSC<I, F> {
        CSC::new(
            self.rows(),
            self.cols(),
            self.rowidx().to_vec(),
            self.colptr().to_vec(),
            self.values().iter().map(|d| d.real()).collect(),
        )
        .unwrap()
    }

    fn imag(&self) -> CSC<I, F> {
        CSC::new(
            self.rows(),
            self.cols(),
            self.rowidx().to_vec(),
            self.colptr().to_vec(),
            self.values().iter().map(|d| d.imag()).collect(),
        )
        .unwrap()
    }

    fn h(&self) -> CSR<I, C> {
        self.conj().transpose()
    }
}
