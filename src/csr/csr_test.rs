use crate::csr::CSR;
use crate::test;

#[test]
fn test_new() -> Result<(), String> {
    let (rowptr, colidx, data) = test::csr_data();

    let csr = CSR::new(test::n, test::n, rowptr, colidx, data)?;

    if csr.rows != test::n {
        return Err(format!("rows, expected {} actual {}", test::n, csr.rows).to_string());
    }
    if csr.cols != test::n {
        return Err(format!("cols, expected {} actual {}", test::n, csr.cols).to_string());
    }

    if csr.nnz() != test::nnz {
        return Err(format!("nnz, expected {} actual {}", test::nnz, csr.nnz()).to_string());
    }

    let diag = test::diagonal();
    for (i, &d) in csr.diagonal().iter().enumerate() {
        if d != diag[i] {
            return Err(format!("diagonal {}, expected {} actual {}", i, diag[i], d).to_string());
        }
    }
    Ok(())
}
