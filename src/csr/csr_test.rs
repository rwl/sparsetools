use crate::csr::CSR;
use crate::test;

#[test]
fn test_new() -> Result<(), String> {
    let (rowptr, colidx, data) = test::csr_data();

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

    test::assert_slice(&csr.diagonal(), &test::diagonal())
}

#[test]
fn test_with_diag() -> Result<(), String> {
    let csr = CSR::<usize, f64>::with_diag(test::diagonal());

    if csr.rows != test::N {
        return Err(format!("rows, expected {} actual {}", test::N, csr.rows).to_string());
    }
    if csr.cols != test::N {
        return Err(format!("cols, expected {} actual {}", test::N, csr.cols).to_string());
    }

    if csr.nnz() != test::N {
        return Err(format!("nnz, expected {} actual {}", test::N, csr.nnz()).to_string());
    }

    test::assert_slice(&csr.diagonal(), &test::diagonal())
}

#[test]
fn test_has_sorted_indexes() -> Result<(), String> {
    {
        let (rowptr, colidx, data) = test::csr_data();

        let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
        if !csr.has_sorted_indexes() {
            return Err("indexes must be sorted".to_string());
        }
    }
    {
        let (rowptr, mut colidx, mut data) = test::csr_data();

        // Swap values on row[1].
        colidx.swap(1, 2);
        data.swap(1, 2);

        let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
        if csr.has_sorted_indexes() {
            return Err("indexes must not be sorted".to_string());
        }
    }
    Ok(())
}
