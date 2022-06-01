use crate::traits::{Integer, Scalar};
use pretty_dtoa::FmtFloatConfig;
use std::collections::HashMap;
use std::io::Write;
use tabwriter::{Alignment, TabWriter};

/// Write A as a sequence of triples.
pub fn csr_string<I: Integer, T: Scalar, W: Write>(
    n_row: I,
    _n_col: I,
    a_p: &[I],
    a_j: &[I],
    a_x: &[T],
    mut w: W,
) -> W {
    for i in 0..n_row.to_usize().unwrap() {
        let start = a_p[i].to_usize().unwrap();
        let end = a_p[i + 1].to_usize().unwrap();

        for jj in start..end {
            writeln!(w, "({}, {}) {}", i, a_j[jj], a_x[jj]).unwrap();
        }
    }
    w
}

/// Write A as a columnar table.
///
/// Input: A is assumed to have canonical format.
/// Rows and columns may optionally be labelled with their index number.
pub fn csr_table<I: Integer, S: Scalar, W: Write>(
    n_row: I,
    n_col: I,
    a_p: &[I],
    a_j: &[I],
    a_x: &[S],
    label: bool,
    w: W,
    fmt_float_config: Option<FmtFloatConfig>,
) -> W {
    let un_row = n_row.to_usize().unwrap();
    let un_col = n_col.to_usize().unwrap();

    let mut tw = TabWriter::new(w)
        .minwidth(5)
        .padding(1)
        .alignment(Alignment::Right);

    if label {
        tw.write(b"\t").unwrap();
        for c in 0..un_col {
            write!(tw, "{}\t", c).unwrap();
        }
        tw.write(b"\n").unwrap();
    }

    for r in 0..un_row {
        if label {
            write!(tw, "{}\t", r).unwrap();
        }

        let start = a_p[r].to_usize().unwrap();
        let end = a_p[r + 1].to_usize().unwrap();

        let mut nz_cols = HashMap::new();
        for jj in start..end {
            nz_cols.insert(a_j[jj].to_usize().unwrap(), jj);
        }

        for c in 0..un_col {
            match nz_cols.get(&c) {
                None => {
                    tw.write(b"-").unwrap();
                }
                Some(&ii) => {
                    let s = a_x[ii].pretty_string(fmt_float_config);
                    tw.write(s.as_bytes()).unwrap();
                }
            }
            if c == un_col - 1 {
                tw.write(b"\t\n").unwrap();
            } else {
                tw.write(b"\t").unwrap();
            }
        }
    }
    tw.into_inner().unwrap()
}
