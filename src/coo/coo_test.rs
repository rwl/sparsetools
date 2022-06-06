use crate::coo::Coo;
use crate::test;

#[test]
fn test_new() -> Result<(), String> {
    let (rowidx, colidx, data) = test::coo_data();
    let coo = Coo::new(test::N, test::N, rowidx, colidx, data)?;
    if coo.rows() != test::N {
        return Err(format!("rows, expected {} actual {}", test::N, coo.rows()));
    }
    if coo.cols() != test::N {
        return Err(format!("cols, expected {} actual {}", test::N, coo.cols()));
    }
    if coo.nnz() != test::NNZ {
        return Err(format!("nnz, expected {} actual {}", test::NNZ, coo.nnz()));
    }
    let diag = test::diagonal();
    for (i, d) in coo.diagonal().into_iter().enumerate() {
        if d != diag[i] {
            return Err(format!("diagonal {}, expected {} actual {}", i, diag[i], d));
        }
    }
    Ok(())
}

#[test]
fn test_empty() -> Result<(), String> {
    let coo = Coo::<usize, f64>::empty(test::N, test::N, test::NNZ);
    if coo.rows() != test::N {
        return Err(format!(
            "empty rows, expected {} actual {}",
            test::N,
            coo.rows()
        ));
    }
    if coo.cols() != test::N {
        return Err(format!(
            "empty cols, expected {} actual {}",
            test::N,
            coo.cols()
        ));
    }
    if coo.nnz() != 0 {
        return Err(format!("empty nnz, expected {} actual {}", 0, coo.nnz()));
    }
    Ok(())
}

#[test]
fn test_identity() -> Result<(), String> {
    let eye = Coo::<usize, f64>::identity(test::N);

    if eye.rows() != test::N {
        return Err(format!(
            "eye rows, expected {} actual {}",
            test::N,
            eye.rows()
        ));
    }
    if eye.cols() != test::N {
        return Err(format!(
            "eye cols, expected {} actual {}",
            test::N,
            eye.cols()
        ));
    }
    if eye.nnz() != test::N {
        return Err(format!(
            "eye nnz, expected {} actual {}",
            test::N,
            eye.nnz()
        ));
    }
    for (i, v) in eye.diagonal().into_iter().enumerate() {
        if v != 1.0 {
            return Err(format!("eye {}, expected {} actual {}", i, 1.0, v));
        }
    }
    Ok(())
}

#[test]
fn test_with_diagonal() -> Result<(), String> {
    let diagonal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let n = diagonal.len();

    let coo = Coo::<usize, f64>::with_diagonal(&diagonal);

    if coo.rows() != n {
        return Err(format!("diag rows, expected {} actual {}", n, coo.rows()));
    }
    if coo.cols() != n {
        return Err(format!("diag cols, expected {} actual {}", n, coo.cols()));
    }
    if coo.nnz() != n {
        return Err(format!("diag nnz, expected {} actual {}", n, coo.nnz()));
    }
    for (i, v) in coo.diagonal().into_iter().enumerate() {
        if v != diagonal[i] {
            return Err(format!("diag {}, expected {} actual {}", i, diagonal[i], v));
        }
    }
    Ok(())
}

#[test]
fn test_transpose() -> Result<(), String> {
    let (rowidx, colidx, data) = test::coo_data();
    let coo0 = Coo::new(test::N, test::N, rowidx, colidx, data)?;
    let coo = coo0.transpose();
    if coo.rows() != test::N {
        return Err(format!(
            "transpose rows, expected {} actual {}",
            test::N,
            coo.rows()
        ));
    }
    if coo.cols() != test::N {
        return Err(format!(
            "transpose cols, expected {} actual {}",
            test::N,
            coo.cols()
        ));
    }
    if coo.nnz() != test::NNZ {
        return Err(format!(
            "transpose nnz, expected {} actual {}",
            test::NNZ,
            coo.nnz()
        ));
    }
    let dense = coo.to_dense();
    if dense[1][2] != 32.0 {
        return Err(format!(
            "dense[{}][{}], expected {} actual {}",
            1, 2, 32.0, dense[1][2]
        ));
    }
    Ok(())
}

#[test]
fn test_add() -> Result<(), String> {
    let (rowidx, colidx, data) = test::coo_data();
    let coo1 = Coo::new(
        test::N,
        test::N,
        rowidx.clone(),
        colidx.clone(),
        data.clone(),
    )?;
    let coo2 = Coo::new(test::N, test::N, rowidx, colidx, data)?;

    let csr = coo1 + coo2.t();

    if csr.rows() != test::N {
        return Err(format!(
            "add rows, expected {} actual {}",
            test::N,
            csr.rows()
        ));
    }
    if csr.cols() != test::N {
        return Err(format!(
            "add cols, expected {} actual {}",
            test::N,
            csr.cols()
        ));
    }
    let diag = test::diagonal();
    for (i, v) in csr.diagonal().into_iter().enumerate() {
        if v != diag[i] + diag[i] {
            return Err(format!(
                "diag {}, expected {} actual {}",
                i,
                diag[i] + diag[i],
                v
            ));
        }
    }
    let dense = csr.to_dense();
    if dense[1][2] != 32.0 {
        return Err(format!(
            "dense[{}][{}], expected {} actual {}",
            1, 2, 32.0, dense[1][2]
        ));
    }
    Ok(())
}

#[test]
fn test_to_dense() -> Result<(), String> {
    let (rowidx, colidx, data) = test::coo_data();
    let coo = Coo::new(test::N, test::N, rowidx, colidx, data)?;

    let dense = coo.to_dense();

    if dense.len() != test::N {
        return Err(format!("rows, expected {} actual {}", test::N, dense.len()));
    }
    for row in &dense {
        if row.len() != test::N {
            return Err(format!("cols, expected {} actual {}", test::N, row.len()));
        }
    }
    if dense[2][1] != 32.0 {
        return Err(format!(
            "dense[{}][{}], expected {} actual {}",
            2, 1, 32.0, dense[2][1]
        ));
    }
    Ok(())
}
