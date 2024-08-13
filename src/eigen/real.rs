use crate::csc::CSC;
use anyhow::{format_err, Result};
use arpack_ng_sys::*;
use full::Mat;
use lazy_static::lazy_static;
use num_complex::Complex64;
use std::sync::Mutex;

lazy_static! {
    static ref MUTEX: Mutex<()> = Mutex::new(());
}

// Standard eigenvalue problem.
const BMAT_STANDARD: &str = "I";

// Want the NEV eigenvalues of largest magnitude.
const WHICH_LARGEST_MAGNITUDE: &str = "LM";

// First call to the reverse communication interface
const IDO_FIRST: i32 = 0;

// Compute  Y = OP * X  where
// IPNTR(1) is the pointer into WORKD for X,
// IPNTR(2) is the pointer into WORKD for Y.
// In mode 3 and 4, the vector B * X is already
// available in WORKD(ipntr(3)).  It does not
// need to be recomputed in forming OP * X.
const IDO_COMPUTE1: i32 = 1;

// Compute NEV Ritz vectors.
const HOW_MNY_RITZ: &str = "A";

pub fn eigs(a_mat: &CSC<usize, f64>, nev: usize) -> Result<(Vec<Complex64>, Mat<Complex64>)> {
    if a_mat.rows() != a_mat.cols() {
        return Err(format_err!(
            "square matrix required ({}, {})",
            a_mat.rows(),
            a_mat.cols()
        ));
    }
    let n = a_mat.rows();

    if nev >= n - 1 {
        return Err(format_err!("invalid nev: {} (0 < nev < {})", nev, n - 1));
    }
    let bmat = BMAT_STANDARD;
    let ncv = usize::min(n, 2 * nev + 1);
    let maxiter = n * 10;

    let tol = 0.0;
    let mut resid = vec![0.0; n];
    let mut info = 0; // random initial vector

    let sigmar = 0.0;
    let sigmai = 0.0;

    let mut v = vec![0.0; n * ncv];
    let ldv = n;

    let mut iparam = vec![0i32; 11];

    iparam[0] = 1;
    iparam[2] = maxiter as i32;
    iparam[3] = 1;
    iparam[6] = 1; // standard

    let which = WHICH_LARGEST_MAGNITUDE;
    let mut ido = IDO_FIRST;

    let mut workd = vec![0.0; 3 * n];
    let lworkl = 3 * ncv * (ncv + 2);
    let mut workl = vec![0.0; lworkl];

    let mut ipntr = vec![0i32; 14];

    unsafe {
        let lock = MUTEX.lock().unwrap();
        dnaupd_c(
            &mut ido,
            bmat.as_ptr() as *const i8,
            n as i32,
            which.as_ptr() as *const i8,
            nev as i32,
            tol,
            resid.as_mut_ptr(),
            ncv as i32,
            v.as_mut_ptr(),
            ldv as i32,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr(),
            workl.as_mut_ptr(),
            lworkl as i32,
            &mut info,
        );
        drop(lock);
    }

    while ido == IDO_COMPUTE1 {
        let y = a_mat.mat_vec(&workd[ipntr[0] as usize - 1..ipntr[0] as usize - 1 + n])?;
        workd[ipntr[1] as usize - 1..].copy_from_slice(&y);

        unsafe {
            let lock = MUTEX.lock().unwrap();
            dnaupd_c(
                &mut ido,
                bmat.as_ptr() as *const i8,
                n as i32,
                which.as_ptr() as *const i8,
                nev as i32,
                tol,
                resid.as_mut_ptr(),
                ncv as i32,
                v.as_mut_ptr(),
                ldv as i32,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                lworkl as i32,
                &mut info,
            );
            drop(lock);
        }
    }
    if info < 0 {
        // return ArpackError(info);
        return Err(format_err!("info = {}", info));
    }

    let rvec = true;
    let howmny = HOW_MNY_RITZ;
    let select = vec![false as i32; ncv];
    let mut dr = vec![0.0; nev + 1];
    let mut di = vec![0.0; nev + 1];
    let mut z = vec![0.0; n * (nev + 1)];
    let ldz = n + 1;
    let mut workev = vec![0.0; 2 * ncv];

    unsafe {
        let lock = MUTEX.lock().unwrap();
        dneupd_c(
            rvec as i32,
            howmny.as_ptr() as *const i8,
            select.as_ptr(),
            dr.as_mut_ptr(),
            di.as_mut_ptr(),
            z.as_mut_ptr(),
            ldz as i32,
            sigmar,
            sigmai,
            workev.as_mut_ptr(),
            bmat.as_ptr() as *const i8,
            n as i32,
            which.as_ptr() as *const i8,
            nev as i32,
            tol,
            resid.as_mut_ptr(),
            ncv as i32,
            v.as_mut_ptr(),
            ldv as i32,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr(),
            workl.as_mut_ptr(),
            lworkl as i32,
            &mut info,
        );
        drop(lock);
    }
    if info != 0 {
        // return ArpackError(info);
        return Err(format_err!("info = {}", info));
    }

    let mut nconv = iparam[4] as usize;

    // let mut d = vec![Complex64::zero(); nev + 1];
    // for i in 0..d.len() {
    //     d[i] = Complex64::new(dr[i], di[i]);
    // }
    let mut d: Vec<Complex64> = (0..nev + 1).map(|i| Complex64::new(dr[i], dr[i])).collect();

    let mut zz = Mat::<Complex64>::zeros(nev + 1, n, true);

    let mut i = 0;
    while i < nev + 1 {
        if di[i].abs() == 0.0 {
            // zz[i] = make([]complex128, n)
            let j = i * n;
            let zr = &z[j..j + n];
            for j in 0..zz.cols() {
                zz[(i, j)] = Complex64::new(zr[i], 0.0);
            }
        } else {
            // The complex Ritz vector associated with the Ritz value
            // with positive imaginary part is stored in two consecutive
            // columns. The first column holds the real part of the Ritz
            // vector and the second column holds the imaginary part. The
            // Ritz vector associated with the Ritz value with negative
            // imaginary part is simply the complex conjugate of the Ritz
            // vector associated with the positive imaginary part.
            if i < nev {
                let j = i * n;
                let zr = &z[j..j + n];
                let j = (i + 1) * n;
                let zi = &z[j..j + n];

                // zz[i] = make([]complex128, n)
                // zz[i+1] = make([]complex128, n)
                // for j := range zz[i] {
                for j in 0..zz.cols() {
                    zz[(i, j)] = Complex64::new(zr[j], zi[j]);
                    zz[(i + 1, j)] = zz[(i, j)].conj();
                }
                i += 1;
            } else {
                // Discard final eigenvalue that has complex part
                // but no accompanying conjugate.
                nconv -= 1; // TODO: return error?
            }
        }
        i += 1;
    }
    if nconv <= nev {
        d = d[..nconv].to_vec();
        // z = z[..nconv].to_vec();
    }

    if info == 1 {
        // let iter = iparam[2];
        // return d, zz, ArpackConvergenceFailure{iter, nconv, nev}
    }
    Ok((d, zz))
}
