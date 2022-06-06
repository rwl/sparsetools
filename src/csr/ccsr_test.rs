use crate::csr::CSR;
use crate::test;

#[test]
fn test_new() -> Result<(), String> {
    let (rowptr, colidx, data) = test::c_csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;

    if csr.rows != test::N {
        return Err(format!("rows, expected {} actual {}", test::N, csr.rows).to_string());
    }
    if csr.cols != test::N {
        return Err(format!("cols, expected {} actual {}", test::N, csr.cols).to_string());
    }

    if csr.nnz() != test::NNZ {
        return Err(format!("nnz, expected {} actual {}", test::NNZ, csr.nnz()).to_string());
    }

    let diag = test::c_diagonal();
    for (i, &d) in csr.diagonal().iter().enumerate() {
        if d != diag[i] {
            return Err(format!("diagonal {}, expected {} actual {}", i, diag[i], d).to_string());
        }
    }
    Ok(())
}
