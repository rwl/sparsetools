use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Complex, Float, Integer};

pub trait CCSR<I, F, C> {
    fn with_real(re: &CSR<I, F>) -> Self;
    fn with_imag(im: &CSR<I, F>) -> Self;
    fn conj(&self) -> Self;
    fn real(&self) -> CSR<I, F>;
    fn imag(&self) -> CSR<I, F>;
    /// Returns the conjugate transpose of self.
    fn h(&self) -> CSC<I, C>;
}

impl<I, F, C> CCSR<I, F, C> for CSR<I, C>
where
    I: Integer,
    F: Float,
    C: Complex<F>,
{
    fn with_real(re: &CSR<I, F>) -> Self {
        Self::new(
            re.rows(),
            re.cols(),
            re.rowptr.clone(),
            re.colidx.clone(),
            re.values.iter().map(|v| C::new(*v, F::zero())).collect(),
        )
        .unwrap()
    }

    fn with_imag(im: &CSR<I, F>) -> Self {
        Self::new(
            im.rows(),
            im.cols(),
            im.rowptr.clone(),
            im.colidx.clone(),
            im.values.iter().map(|v| C::new(F::zero(), *v)).collect(),
        )
        .unwrap()
    }

    fn conj(&self) -> CSR<I, C> {
        CSR::new(
            self.rows(),
            self.cols(),
            self.rowptr().to_vec(),
            self.colidx().to_vec(),
            self.values().iter().map(|d| d.conj()).collect(),
        )
        .unwrap()
    }

    fn real(&self) -> CSR<I, F> {
        CSR::new(
            self.rows(),
            self.cols(),
            self.rowptr().to_vec(),
            self.colidx().to_vec(),
            self.values().iter().map(|d| d.real()).collect(),
        )
        .unwrap()
    }

    fn imag(&self) -> CSR<I, F> {
        CSR::new(
            self.rows(),
            self.cols(),
            self.rowptr().to_vec(),
            self.colidx().to_vec(),
            self.values().iter().map(|d| d.imag()).collect(),
        )
        .unwrap()
    }

    fn h(&self) -> CSC<I, C> {
        self.conj().transpose()
    }
}
