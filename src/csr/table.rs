use crate::traits::{Integer, Scalar};
use pretty_dtoa::FmtFloatConfig;
use std::collections::HashMap;
use std::io::Write;
use tabwriter::{Alignment, TabWriter};

/// Write A as a columnar table.
///
/// Input: A is assumed to have canonical format.
/// Rows and columns may optionally be labelled with their index number.
pub fn csr_table<I: Integer, S: Scalar, W: Write>(
    n_row: usize,
    n_col: usize,
    a_p: &[I],
    a_j: &[I],
    a_x: &[S],
    label: bool,
    w: W,
    fmt_float_config: Option<FmtFloatConfig>,
) -> W {
    let mut tw = TabWriter::new(w)
        .minwidth(5)
        .padding(1)
        .alignment(Alignment::Right);

    if label {
        tw.write(b"\t").unwrap();
        for c in 0..n_col {
            write!(tw, "{}\t", c).unwrap();
        }
        tw.write(b"\n").unwrap();
    }

    for r in 0..n_row {
        if label {
            write!(tw, "{}\t", r).unwrap();
        }

        let start = a_p[r].to_usize().unwrap();
        let end = a_p[r + 1].to_usize().unwrap();

        let mut nz_cols = HashMap::new();
        for jj in start..end {
            nz_cols.insert(a_j[jj].to_usize().unwrap(), jj);
        }

        for c in 0..n_col {
            match nz_cols.get(&c) {
                None => {
                    tw.write(b"-").unwrap();
                }
                Some(&ii) => {
                    let s = a_x[ii].pretty_string(fmt_float_config);
                    tw.write(s.as_bytes()).unwrap();
                }
            }
            if c == n_col - 1 {
                tw.write(b"\t\n").unwrap();
            } else {
                tw.write(b"\t").unwrap();
            }
        }
    }
    tw.into_inner().unwrap()
}
