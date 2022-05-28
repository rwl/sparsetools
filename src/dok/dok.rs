use crate::coo::Coo;
use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use std::collections::HashMap;
use std::hash::Hash;

/// A sparse matrix with scalar values stored in Dictionary of Keys format.
pub struct DoK<I, T> {
    pub rows: I,
    pub cols: I,
    /// Explicitly stored values (size nnz).
    pub data: HashMap<(I, I), T>,
}

impl<I: Integer + Hash, T: Scalar> DoK<I, T> {
    pub fn new(rows: I, cols: I, capacity: I) -> Self {
        Self {
            rows,
            cols,
            data: HashMap::with_capacity(capacity.to_usize().unwrap()),
        }
    }

    pub fn from_dense(d: &[&[T]]) -> Self {
        let m = d.len();
        let n = if m > 0 { d[0].len() } else { 0 };

        let mut mat = Self::new(
            I::from(m).unwrap(),
            I::from(n).unwrap(),
            I::from(m * n).unwrap(),
        );

        for (i, &row) in d.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != T::zero() {
                    mat.set(I::from(i).unwrap(), I::from(j).unwrap(), v)
                        .unwrap();
                }
            }
        }
        mat
    }

    pub fn nnz(&self) -> I {
        I::from(self.data.len()).unwrap()
    }

    pub fn get(&self, i: I, j: I) -> T {
        match self.data.get(&(i, j)) {
            None => T::zero(),
            Some(&v) => v,
        }
    }

    pub fn set(&mut self, i: I, j: I, v: T) -> Result<(), String> {
        if i < I::zero() || i >= self.rows {
            return Err(format!("row index {} out of range {}", i, self.rows).to_string());
        }
        if j < I::zero() || j >= self.cols {
            return Err(format!("col index {} out of range {}", j, self.cols).to_string());
        }
        if v == T::zero() {
            self.data.remove(&(i, j));
        } else {
            self.data.insert((i, j), v);
        }
        Ok(())
    }

    pub fn add(&mut self, i: I, j: I, v: T) -> Result<(), String> {
        self.set(i, j, self.get(i, j) + v)
    }

    pub fn sub(&mut self, i: I, j: I, v: T) -> Result<(), String> {
        self.set(i, j, self.get(i, j) - v)
    }

    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let rows = self.rows.to_usize().unwrap();
        let cols = self.cols.to_usize().unwrap();
        let mut d = vec![vec![T::zero(); cols]; rows];
        for (&(i, j), &v) in &self.data {
            d[i.to_usize().unwrap()][j.to_usize().unwrap()] = v;
        }
        d
    }

    pub fn to_coo(&self) -> Coo<I, T> {
        let nnz = self.data.len();

        let mut rowidx = vec![I::zero(); nnz];
        let mut colidx = vec![I::zero(); nnz];
        let mut data = vec![T::zero(); nnz];

        for (k, (&(i, j), &v)) in self.data.iter().enumerate() {
            rowidx[k] = i;
            colidx[k] = j;
            data[k] = v;
        }

        Coo {
            rows: self.rows,
            cols: self.cols,
            rowidx,
            colidx,
            data,
        }
    }

    pub fn to_csc(&self) -> CSC<I, T> {
        self.to_coo().to_csc()
    }

    pub fn to_csr(&self) -> CSR<I, T> {
        self.to_coo().to_csr()
    }

    pub fn to_string(&self) -> String {
        let mut buf: String = String::new();
        for (&(i, j), &v) in &self.data {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&format!("({}, {}) {}", i, j, v));
        }
        buf
    }
}
