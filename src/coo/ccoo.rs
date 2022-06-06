use crate::coo::Coo;
use crate::traits::{Complex, Float, Integer};

pub trait CCoo<I, F> {
    fn conj(&self) -> Self;
    fn real(&self) -> Coo<I, F>;
    fn imag(&self) -> Coo<I, F>;
}

impl<I, F, C> CCoo<I, F> for Coo<I, C>
where
    I: Integer,
    F: Float,
    C: Complex<F>,
{
    fn conj(&self) -> Coo<I, C> {
        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx: self.rowidx.clone(),
            colidx: self.colidx.clone(),
            data: self.data.iter().map(|d| d.conj()).collect(),
        }
    }

    fn real(&self) -> Coo<I, F> {
        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx: self.rowidx.clone(),
            colidx: self.colidx.clone(),
            data: self.data.iter().map(|d| d.real()).collect(),
        }
    }

    fn imag(&self) -> Coo<I, F> {
        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx: self.rowidx.clone(),
            colidx: self.colidx.clone(),
            data: self.data.iter().map(|d| d.imag()).collect(),
        }
    }
}
