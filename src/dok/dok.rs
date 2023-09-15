use crate::coo::Coo;
use crate::csc::CSC;
use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use anyhow::{format_err, Result};
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;

/// A sparse matrix with scalar values stored in Dictionary of Keys format.
pub struct DoK<I, T> {
    rows: usize,
    cols: usize,
    data: HashMap<(I, I), T>,
}

impl<I: Integer + Hash, T: Scalar> DoK<I, T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: HashMap::new(),
        }
    }

    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        Self {
            rows,
            cols,
            data: HashMap::with_capacity(capacity),
        }
    }

    pub fn from_dense(d: &[&[T]]) -> Self {
        let m = d.len();
        let n = if m > 0 { d[0].len() } else { 0 };

        let mut mat = Self::with_capacity(m, n, m * n);

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

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, i: I, j: I) -> T {
        match self.data.get(&(i, j)) {
            None => T::zero(),
            Some(&v) => v,
        }
    }

    pub fn set(&mut self, i: I, j: I, v: T) -> Result<()> {
        if i < I::zero() || i >= I::from(self.rows).unwrap() {
            return Err(format_err!("row index {} out of range {}", i, self.rows));
        }
        if j < I::zero() || j >= I::from(self.cols).unwrap() {
            return Err(format_err!("col index {} out of range {}", j, self.cols));
        }
        if v == T::zero() {
            self.data.remove(&(i, j));
        } else {
            self.data.insert((i, j), v);
        }
        Ok(())
    }

    pub fn add(&mut self, i: I, j: I, v: T) -> Result<()> {
        self.set(i, j, self.get(i, j) + v)
    }

    pub fn sub(&mut self, i: I, j: I, v: T) -> Result<()> {
        self.set(i, j, self.get(i, j) - v)
    }

    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let mut d = vec![vec![T::zero(); self.cols]; self.rows];
        for (&(i, j), &v) in &self.data {
            d[i.to_usize().unwrap()][j.to_usize().unwrap()] = v;
        }
        d
    }

    pub fn to_coo(&self) -> Coo<I, T> {
        let nnz = self.data.len();

        let mut rowidx = Vec::with_capacity(nnz);
        let mut colidx = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);

        for (&(i, j), &v) in &self.data {
            rowidx.push(i);
            colidx.push(j);
            data.push(v);
        }

        Coo::new(self.rows, self.cols, rowidx, colidx, data).unwrap()
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

impl<I, T> Deref for DoK<I, T> {
    type Target = HashMap<(I, I), T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
