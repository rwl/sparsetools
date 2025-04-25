use crate::coo::Coo;
use crate::{Integer, Scalar};

pub struct CooIterator<'a, I, T> {
    coo: &'a Coo<I, T>,
    index: usize,
}

impl<'a, I: Integer, T: Scalar> Iterator for CooIterator<'a, I, T> {
    type Item = (&'a I, &'a I, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.coo.nnz() {
            let result = Some((
                &self.coo.rowidx[self.index],
                &self.coo.colidx[self.index],
                &self.coo.values[self.index],
            ));
            self.index += 1;
            result
        } else {
            None
        }
    }
}

impl<I, T> Coo<I, T> {
    pub fn iter(&self) -> CooIterator<I, T> {
        CooIterator {
            coo: self,
            index: 0,
        }
    }
}

pub struct CooIntoIterator<I, T> {
    coo: Coo<I, T>,
}

impl<I: Integer, T: Scalar> Iterator for CooIntoIterator<I, T> {
    type Item = (I, I, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.coo.nnz() == 0 {
            return None;
        }
        let result = (
            self.coo.rowidx.remove(0),
            self.coo.colidx.remove(0),
            self.coo.values.remove(0),
        );
        Some(result)
    }
}

impl<I: Integer, T: Scalar> IntoIterator for Coo<I, T> {
    type Item = (I, I, T);
    type IntoIter = CooIntoIterator<I, T>;

    fn into_iter(self) -> CooIntoIterator<I, T> {
        CooIntoIterator { coo: self }
    }
}
