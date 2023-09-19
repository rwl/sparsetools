use crate::coo::Coo;
use crate::test;
use anyhow::{format_err, Result};

#[test]
fn test_new() -> Result<()> {
    let (rowidx, colidx, values) = test::coo_data();
    let coo = Coo::new(test::N, test::N, rowidx, colidx, values)?;
    if coo.rows() != test::N {
        return Err(format_err!(
            "rows, expected {} actual {}",
            test::N,
            coo.rows()
        ));
    }
    if coo.cols() != test::N {
        return Err(format_err!(
            "cols, expected {} actual {}",
            test::N,
            coo.cols()
        ));
    }
    if coo.nnz() != test::NNZ {
        return Err(format_err!(
            "nnz, expected {} actual {}",
            test::NNZ,
            coo.nnz()
        ));
    }

    test::assert_slice(&coo.diagonal(), &test::diagonal())?;

    Ok(())
}

#[test]
fn test_empty() -> Result<()> {
    let coo = Coo::<usize, f64>::with_size(test::N, test::N);
    if coo.rows() != test::N {
        return Err(format_err!(
            "empty rows, expected {} actual {}",
            test::N,
            coo.rows()
        ));
    }
    if coo.cols() != test::N {
        return Err(format_err!(
            "empty cols, expected {} actual {}",
            test::N,
            coo.cols()
        ));
    }
    if coo.nnz() != 0 {
        return Err(format_err!(
            "empty nnz, expected {} actual {}",
            0,
            coo.nnz()
        ));
    }

    test::assert_slice(&coo.diagonal(), &vec![0.0; test::N])?;

    Ok(())
}

#[test]
fn test_identity() -> Result<()> {
    let eye = Coo::<usize, f64>::identity(test::N);

    if eye.rows() != test::N {
        return Err(format_err!(
            "eye rows, expected {} actual {}",
            test::N,
            eye.rows()
        ));
    }
    if eye.cols() != test::N {
        return Err(format_err!(
            "eye cols, expected {} actual {}",
            test::N,
            eye.cols()
        ));
    }
    if eye.nnz() != test::N {
        return Err(format_err!(
            "eye nnz, expected {} actual {}",
            test::N,
            eye.nnz()
        ));
    }

    test::assert_slice(&eye.diagonal(), &vec![1.0; test::N])?;

    Ok(())
}

#[test]
fn test_with_diagonal() -> Result<()> {
    let diagonal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let n = diagonal.len();

    let coo = Coo::<usize, f64>::with_diagonal(&diagonal);

    if coo.rows() != n {
        return Err(format_err!(
            "diag rows, expected {} actual {}",
            n,
            coo.rows()
        ));
    }
    if coo.cols() != n {
        return Err(format_err!(
            "diag cols, expected {} actual {}",
            n,
            coo.cols()
        ));
    }
    if coo.nnz() != n {
        return Err(format_err!("diag nnz, expected {} actual {}", n, coo.nnz()));
    }

    test::assert_slice(&coo.diagonal(), &diagonal)?;

    Ok(())
}

#[test]
fn test_transpose() -> Result<()> {
    let (rowidx, colidx, values) = test::coo_data();
    let coo0 = Coo::new(test::N, test::N, rowidx, colidx, values)?;
    let coo = coo0.transpose();
    if coo.rows() != test::N {
        return Err(format_err!(
            "transpose rows, expected {} actual {}",
            test::N,
            coo.rows()
        ));
    }
    if coo.cols() != test::N {
        return Err(format_err!(
            "transpose cols, expected {} actual {}",
            test::N,
            coo.cols()
        ));
    }
    if coo.nnz() != test::NNZ {
        return Err(format_err!(
            "transpose nnz, expected {} actual {}",
            test::NNZ,
            coo.nnz()
        ));
    }
    let dense = coo.to_dense();
    if dense[1][2] != 32.0 {
        return Err(format_err!(
            "dense[{}][{}], expected {} actual {}",
            1,
            2,
            32.0,
            dense[1][2]
        ));
    }
    Ok(())
}

#[test]
fn test_add() -> Result<()> {
    let (rowidx, colidx, values) = test::coo_data();
    let coo1 = Coo::new(
        test::N,
        test::N,
        rowidx.clone(),
        colidx.clone(),
        values.clone(),
    )?;
    let coo2 = Coo::new(test::N, test::N, rowidx, colidx, values)?;

    let csr = coo1 + coo2.t();

    if csr.rows() != test::N {
        return Err(format_err!(
            "add rows, expected {} actual {}",
            test::N,
            csr.rows()
        ));
    }
    if csr.cols() != test::N {
        return Err(format_err!(
            "add cols, expected {} actual {}",
            test::N,
            csr.cols()
        ));
    }
    let diag = test::diagonal();
    for (i, v) in csr.diagonal().into_iter().enumerate() {
        if v != diag[i] + diag[i] {
            return Err(format_err!(
                "diag {}, expected {} actual {}",
                i,
                diag[i] + diag[i],
                v
            ));
        }
    }
    let dense = csr.to_dense();
    if dense[1][2] != 32.0 {
        return Err(format_err!(
            "dense[{}][{}], expected {} actual {}",
            1,
            2,
            32.0,
            dense[1][2]
        ));
    }
    Ok(())
}

#[test]
fn test_to_dense() -> Result<()> {
    let (rowidx, colidx, values) = test::coo_data();
    let coo = Coo::new(test::N, test::N, rowidx, colidx, values)?;

    test::assert_dense(&coo.to_dense(), &test::dense_data())?;
    Ok(())
}

#[test]
fn test_to_csc() -> Result<()> {
    let (rowidx, colidx, values) = test::coo_data();
    let coo = Coo::new(test::N, test::N, rowidx, colidx, values)?;
    let csc = coo.to_csc();

    if csc.rows() != test::N {
        return Err(format_err!(
            "to_csc rows, expected {} actual {}",
            test::N,
            csc.rows()
        ));
    }
    if csc.cols() != test::N {
        return Err(format_err!(
            "to_csc cols, expected {} actual {}",
            test::N,
            csc.cols()
        ));
    }

    let (rowidx, colptr, values) = test::csc_data();
    test::assert_slice(&csc.rowidx(), &rowidx)?;
    test::assert_slice(&csc.colptr(), &colptr)?;
    test::assert_slice(&csc.values(), &values)?;

    Ok(())
}

#[test]
fn test_to_csr_sum_duplicates() -> Result<()> {
    let (rowidx0, colidx0, values0) = test::coo_data();

    let rowidx = [rowidx0.clone(), rowidx0.clone()].concat();
    let colidx = [colidx0.clone(), colidx0.clone()].concat();
    let values = [values0.clone(), values0.clone()].concat();

    let coo = Coo::new(test::N, test::N, rowidx, colidx, values)?;
    let csr = coo.to_csr();

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

    test::assert_dense(
        &csr.to_dense(),
        &test::dense_data()
            .iter()
            .map(|row| row.iter().map(|&v| v + v).collect())
            .collect(),
    )?;

    Ok(())
}

#[test]
fn test_v_stack() -> Result<()> {
    let (rowidx1, colidx1, values1) = test::coo_data();
    let coo1 = Coo::new(test::N, test::N, rowidx1, colidx1, values1)?;

    let (rowidx2, colidx2, values2) = test::coo_data();
    let coo2 = Coo::new(test::N, test::N, rowidx2, colidx2, values2)?;

    let coo = Coo::v_stack(&coo1, &coo2)?;

    test::assert_dense(
        &coo.to_dense(),
        &[test::dense_data(), test::dense_data()].concat(),
    )?;
    Ok(())
}

#[test]
fn test_h_stack() -> Result<()> {
    let (rowidx1, colidx1, values1) = test::coo_data();
    let coo1 = Coo::new(test::N, test::N, rowidx1, colidx1, values1)?;

    let (rowidx2, colidx2, values2) = test::coo_data();
    let coo2 = Coo::new(test::N, test::N, rowidx2, colidx2, values2)?;

    let coo = Coo::h_stack(&coo1, &coo2)?;

    test::assert_dense(
        &coo.to_dense(),
        &test::dense_data()
            .iter()
            .map(|row| [row.clone(), row.clone()].concat())
            .collect(),
    )?;
    Ok(())
}
