use crate::csr::CSR;
use crate::test;
use anyhow::{format_err, Result};

#[test]
fn test_new() -> Result<()> {
    let (rowptr, colidx, values) = test::c_csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, values)?;

    if csr.rows() != test::N {
        return Err(format_err!(
            "rows, expected {} actual {}",
            test::N,
            csr.rows()
        ));
    }
    if csr.cols() != test::N {
        return Err(format_err!(
            "cols, expected {} actual {}",
            test::N,
            csr.cols()
        ));
    }

    if csr.nnz() != test::NNZ {
        return Err(format_err!(
            "nnz, expected {} actual {}",
            test::NNZ,
            csr.nnz()
        ));
    }

    let diag = test::c_diagonal();
    for (i, &d) in csr.diagonal().iter().enumerate() {
        if d != diag[i] {
            return Err(format_err!(
                "diagonal {}, expected {} actual {}",
                i,
                diag[i],
                d
            ));
        }
    }
    Ok(())
}
