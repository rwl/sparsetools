use crate::coo::Coo;
use crate::traits::{Complex, Float, Integer};

pub trait CCoo<I, F> {
    fn conj(&self) -> Self;
    fn real(&self) -> Coo<I, F>;
    fn imag(&self) -> Coo<I, F>;
    /// Returns the conjugate transpose of self.
    fn h(&self) -> Self;
}

impl<I, F, C> CCoo<I, F> for Coo<I, C>
where
    I: Integer,
    F: Float,
    C: Complex<F>,
{
    fn conj(&self) -> Coo<I, C> {
        Coo::new(
            self.rows(),
            self.cols(),
            self.rowidx().to_vec(),
            self.colidx().to_vec(),
            self.data().iter().map(|d| d.conj()).collect::<Vec<C>>(),
        )
        .unwrap()
    }

    fn real(&self) -> Coo<I, F> {
        Coo::new(
            self.rows(),
            self.cols(),
            self.rowidx().to_vec(),
            self.colidx().to_vec(),
            self.data().iter().map(|d| d.real()).collect(),
        )
        .unwrap()
    }

    fn imag(&self) -> Coo<I, F> {
        Coo::new(
            self.rows(),
            self.cols(),
            self.rowidx().to_vec(),
            self.colidx().to_vec(),
            self.data().iter().map(|d| d.imag()).collect(),
        )
        .unwrap()
    }

    fn h(&self) -> Self {
        self.conj().t()
    }
}
