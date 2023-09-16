use crate::coo::Coo;
use crate::csr::CSR;
use crate::test;
use crate::test::csr_data;
use anyhow::{format_err, Result};

#[test]
fn test_new() -> Result<()> {
    let (rowptr, colidx, data) = test::csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;

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

    test::assert_slice(&csr.diagonal(), &test::diagonal())
}

#[test]
fn test_with_diag() -> Result<()> {
    let csr = CSR::<usize, f64>::with_diagonal(test::diagonal());

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

    if csr.nnz() != test::N {
        return Err(format_err!(
            "nnz, expected {} actual {}",
            test::N,
            csr.nnz()
        ));
    }

    test::assert_slice(&csr.diagonal(), &test::diagonal())
}

#[test]
fn test_has_sorted_indexes() -> Result<()> {
    {
        let (rowptr, colidx, data) = test::csr_data();

        let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
        if !csr.has_sorted_indexes() {
            return Err(format_err!("indexes must be sorted"));
        }
    }
    {
        let (rowptr, mut colidx, mut data) = test::csr_data();

        // Swap values on row[1].
        colidx.swap(1, 2);
        data.swap(1, 2);

        let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
        if csr.has_sorted_indexes() {
            return Err(format_err!("indexes must not be sorted"));
        }
    }
    Ok(())
}

#[test]
fn test_sort_indexes() -> Result<()> {
    let (rowptr, mut colidx, mut data) = test::csr_data();

    // Swap values on row[1].
    colidx.swap(1, 2);
    data.swap(1, 2);

    let mut csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
    csr.sort_indexes();
    if !csr.has_sorted_indexes() {
        return Err(format_err!("indexes must be sorted"));
    }
    Ok(())
}

#[test]
fn test_to_coo() -> Result<()> {
    let (rowptr, colidx, data) = test::csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
    let coo = csr.to_coo();
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
    for (i, (&act, exp)) in coo.diagonal().iter().zip(test::diagonal()).enumerate() {
        if act != exp {
            return Err(format_err!(
                "diagonal {}, expected {} actual {}",
                i,
                exp,
                act
            ));
        }
    }

    let dense = coo.to_dense();
    let (act, exp) = (dense[2][1], 32.0);
    if act != exp {
        return Err(format_err!(
            "dense[{}][{}], expected {} actual {}",
            2,
            1,
            exp,
            act
        ));
    }
    Ok(())
}

#[test]
fn test_to_csc() -> Result<()> {
    let (rowptr, colidx, data) = test::csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
    let csc = csr.to_csc();
    if csc.rows() != test::N {
        return Err(format_err!(
            "rows, expected {} actual {}",
            test::N,
            csc.rows()
        ));
    }
    if csc.cols() != test::N {
        return Err(format_err!(
            "cols, expected {} actual {}",
            test::N,
            csc.cols()
        ));
    }
    if csc.nnz() != test::NNZ {
        return Err(format_err!(
            "nnz, expected {} actual {}",
            test::NNZ,
            csc.nnz()
        ));
    }
    for (i, (&act, exp)) in csc.diagonal().iter().zip(test::diagonal()).enumerate() {
        if act != exp {
            return Err(format_err!(
                "diagonal {}, expected {} actual {}",
                i,
                exp,
                act
            ));
        }
    }
    Ok(())
}

#[test]
fn test_transpose() -> Result<()> {
    let a = vec![vec![1.0, 2.0, 3.0], vec![0.0, 2.0, 0.0]];
    let csr = Coo::<usize, f64>::from_dense(&a).to_csr();
    let csc = csr.clone().transpose();

    // println!("\nCSR:\n{}", csr.to_table());
    // println!("\nCSC:\n{}", csc.to_csr().table());

    if csc.rows() != csr.cols() {
        return Err(format_err!(
            "transpose rows, expected {} actual {}",
            csr.cols(),
            csc.rows()
        ));
    }
    if csc.cols() != csr.rows() {
        return Err(format_err!(
            "transpose cols, expected {} actual {}",
            csr.rows(),
            csc.cols()
        ));
    }
    if csc.nnz() != csr.nnz() {
        return Err(format_err!(
            "transpose nnz, expected {} actual {}",
            csr.nnz(),
            csc.nnz()
        ));
    }
    // if &csc.Data()[0] != &csr.Data()[0] {
    // 	t.Error("transpose, data must not be copied")
    // }
    for (i, row) in csc.to_dense().iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != a[j][i] {
                return Err(format_err!(
                    "transpose {},{}, expected {} actual {}",
                    i,
                    j,
                    a[j][i],
                    v
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn test_mat_vec() -> Result<()> {
    let a = vec![vec![1.0, 2.0, 3.0], vec![0.0, 2.0, 0.0]];
    let csr = Coo::<usize, f64>::from_dense(&a).to_csr();
    let b = csr.mat_vec(&vec![1.0, 2.0, 3.0])?;

    let c = vec![14.0, 4.0];
    if b.len() != c.len() {
        return Err(format_err!(
            "matvec len, expected {} actual {}",
            c.len(),
            b.len()
        ));
    }
    for i in 0..c.len() {
        if b[i] != c[i] {
            return Err(format_err!(
                "matvec {}, expected {} actual {}",
                i,
                c[i],
                b[i]
            ));
        }
    }

    Ok(())
}

#[test]
fn test_mat_mat() -> Result<()> {
    let a = Coo::<usize, f64>::from_dense(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).to_csr();
    let b = Coo::<usize, f64>::from_dense(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
        .to_csr();

    let c = a.mat_mat(&b)?;
    if c.rows() != a.rows() || c.cols() != b.cols() {
        return Err(format_err!("matmat, result dimensions"));
    }

    let d = c.to_dense();
    for (i, row) in vec![vec![22.0, 28.0], vec![49.0, 64.0]].iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if d[i][j] != v {
                return Err(format_err!(
                    "csr matmat {},{}, expected {} actual {}",
                    i,
                    j,
                    v,
                    d[i][j]
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn test_select() -> Result<()> {
    let (rowptr, colidx, data) = csr_data();

    let csr = CSR::new(test::N, test::N, rowptr, colidx, data)?;
    let rowidx = vec![1, 2, 3];
    let colidx = vec![0, 1, 3];

    let sel = csr.select(Some(&rowidx), Some(&colidx))?;
    if sel.rows() != rowidx.len() {
        return Err(format_err!(
            "rows, expected {} actual {}",
            rowidx.len(),
            sel.rows()
        ));
    }
    if sel.cols() != colidx.len() {
        return Err(format_err!(
            "cols, expected {} actual {}",
            colidx.len(),
            sel.cols()
        ));
    }
    if sel.nnz() != 5 {
        return Err(format_err!("nnz, expected {} actual {}", 5, sel.nnz()));
    }

    let diag = vec![21.0, 32.0, 44.0];
    for (i, &d) in sel.diagonal().iter().enumerate() {
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

#[test]
fn test_connected_components() -> Result<()> {
    let n: usize = 12;
    let mut coo = Coo::with_size(n, n);

    let mut push_sym = |i, j| {
        coo.push(i, j, 1.0);
        coo.push(j, i, 1.0);
    };

    push_sym(1, 2);
    push_sym(2, 3);

    push_sym(4, 6);
    push_sym(6, 8);

    push_sym(7, 10);
    push_sym(11, 10);
    // coo.push(1, 2, 1.0);
    // coo.push(2, 3, 1.0);
    //
    // coo.push(4, 6, 1.0);
    // coo.push(6, 8, 1.0);
    //
    // coo.push(7, 10, 1.0);
    // coo.push(11, 10, 1.0);

    // println!("{}", coo.to_csr().to_table());

    let (ncc, flags) = coo.to_csr().connected_components::<isize>()?;
    println!("{:?}", flags);
    assert_eq!(ncc, 3);

    assert!(flags[0] < 0);
    assert_eq!(flags[1], 0);
    assert_eq!(flags[2], 0);
    assert_eq!(flags[3], 0);
    assert_eq!(flags[4], 1);
    assert!(flags[5] < 0);
    assert_eq!(flags[6], 1);
    assert_eq!(flags[7], 2);
    assert_eq!(flags[8], 1);
    assert!(flags[9] < 0);
    assert_eq!(flags[10], 2);
    assert_eq!(flags[11], 2);

    Ok(())
}
