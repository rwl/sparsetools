use crate::coo::Coo;
use crate::csr::CSR;
use crate::test;
use crate::test::csr_data;

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

#[test]
fn test_sort_indexes() -> Result<(), String> {
    let (rowptr, mut colidx, mut data) = test::csr_data();

    // Swap values on row[1].
    colidx.swap(1, 2);
    data.swap(1, 2);

    let mut csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
    csr.sort_indexes();
    if !csr.has_sorted_indexes() {
        return Err("indexes must be sorted".to_string());
    }
    Ok(())
}

#[test]
fn test_to_coo() -> Result<(), String> {
    let (rowptr, colidx, data) = test::csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
    let coo = csr.to_coo();
    if coo.rows != test::N {
        return Err(format!("rows, expected {} actual {}", test::N, coo.rows).to_string());
    }
    if coo.cols != test::N {
        return Err(format!("cols, expected {} actual {}", test::N, coo.cols).to_string());
    }
    if coo.nnz() != test::NNZ {
        return Err(format!("nnz, expected {} actual {}", test::NNZ, coo.nnz()).to_string());
    }
    for (i, (&act, exp)) in coo.diagonal().iter().zip(test::diagonal()).enumerate() {
        if act != exp {
            return Err(format!("diagonal {}, expected {} actual {}", i, exp, act).to_string());
        }
    }

    let dense = coo.to_dense();
    let (act, exp) = (dense[2][1], 32.0);
    if act != exp {
        return Err(format!("dense[{}][{}], expected {} actual {}", 2, 1, exp, act).to_string());
    }
    Ok(())
}

#[test]
fn test_to_csc() -> Result<(), String> {
    let (rowptr, colidx, data) = test::csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
    let csc = csr.to_csc();
    if csc.rows != test::N {
        return Err(format!("rows, expected {} actual {}", test::N, csc.rows).to_string());
    }
    if csc.cols != test::N {
        return Err(format!("cols, expected {} actual {}", test::N, csc.cols).to_string());
    }
    if csc.nnz() != test::NNZ {
        return Err(format!("nnz, expected {} actual {}", test::NNZ, csc.nnz()).to_string());
    }
    for (i, (&act, exp)) in csc.diagonal().iter().zip(test::diagonal()).enumerate() {
        if act != exp {
            return Err(format!("diagonal {}, expected {} actual {}", i, exp, act).to_string());
        }
    }
    Ok(())
}

#[test]
fn test_transpose() -> Result<(), String> {
    let a = vec![vec![1.0, 2.0, 3.0], vec![0.0, 2.0, 0.0]];
    let csr = Coo::<usize, f64>::from_dense(&a).to_csr();
    let csc = csr.clone().transpose();

    println!("\nCSR:\n{}", csr.to_table());
    // println!("\nCSC:\n{}", csc.to_csr().table());

    if csc.rows != csr.cols {
        return Err(
            format!("transpose rows, expected {} actual {}", csr.cols, csc.rows).to_string(),
        );
    }
    if csc.cols != csr.rows {
        return Err(
            format!("transpose cols, expected {} actual {}", csr.rows, csc.cols).to_string(),
        );
    }
    if csc.nnz() != csr.nnz() {
        return Err(
            format!("transpose nnz, expected {} actual {}", csr.nnz(), csc.nnz()).to_string(),
        );
    }
    // if &csc.Data()[0] != &csr.Data()[0] {
    // 	t.Error("transpose, data must not be copied")
    // }
    for (i, row) in csc.to_dense().iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != a[j][i] {
                return Err(
                    format!("transpose {},{}, expected {} actual {}", i, j, a[j][i], v).to_string(),
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_mat_vec() -> Result<(), String> {
    let a = vec![vec![1.0, 2.0, 3.0], vec![0.0, 2.0, 0.0]];
    let csr = Coo::<usize, f64>::from_dense(&a).to_csr();
    let b = csr.mat_vec(&vec![1.0, 2.0, 3.0])?;

    let c = vec![14.0, 4.0];
    if b.len() != c.len() {
        return Err(format!(
            "matvec len, expected {} actual {}",
            c.len(),
            b.len()
        ));
    }
    for i in 0..c.len() {
        if b[i] != c[i] {
            return Err(format!("matvec {}, expected {} actual {}", i, c[i], b[i]));
        }
    }

    Ok(())
}

#[test]
fn test_mat_mat() -> Result<(), String> {
    let a = Coo::<usize, f64>::from_dense(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).to_csr();
    let b = Coo::<usize, f64>::from_dense(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
        .to_csr();

    let c = a.mat_mat(&b)?;
    if c.rows() != a.rows() || c.cols() != b.cols() {
        return Err("matmat, result dimensions".to_string());
    }

    let d = c.to_dense();
    for (i, row) in vec![vec![22.0, 28.0], vec![49.0, 64.0]].iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if d[i][j] != v {
                return Err(
                    format!("csr matmat {},{}, expected {} actual {}", i, j, v, d[i][j])
                        .to_string(),
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_select() -> Result<(), String> {
    let (rowptr, colidx, data) = csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
    let rowidx = vec![1, 2, 3];
    let colidx = vec![0, 1, 3];

    let sel = csr.select(Some(&rowidx), Some(&colidx))?;
    if sel.rows() != rowidx.len() {
        return Err(format!("rows, expected {} actual {}", rowidx.len(), sel.rows()).to_string());
    }
    if sel.cols() != colidx.len() {
        return Err(format!("cols, expected {} actual {}", colidx.len(), sel.cols()).to_string());
    }
    if sel.nnz() != 5 {
        return Err(format!("nnz, expected {} actual {}", 5, sel.nnz()).to_string());
    }

    let diag = vec![21.0, 32.0, 44.0];
    for (i, &d) in sel.diagonal().iter().enumerate() {
        if d != diag[i] {
            return Err(format!("diagonal {}, expected {} actual {}", i, diag[i], d).to_string());
        }
    }

    Ok(())
}
