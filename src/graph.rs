use crate::traits::{Integer, Scalar};

/// Determine connected compoments of a compressed sparse graph.
///
/// Note: Output array flag must be preallocated
pub fn cs_graph_components<I: Integer, T: Scalar, F: Integer + num_traits::Signed>(
    n_nod: I,
    a_p: &[I],
    a_j: &[I],
    flag: &mut [F],
) -> Result<I, String> {
    let un_nod = n_nod.to_usize().unwrap();

    // pos is a work array: list of nodes (rows) to process.
    // std::vector<I> pos(n_nod,01);
    let mut pos = vec![I::one(); un_nod];
    let mut n_comp = I::zero();

    let mut n_stop = un_nod;
    for ir in 0..un_nod {
        flag[ir] = F::from(-1).unwrap();
        if a_p[ir + 1] == a_p[ir] {
            n_stop -= 1;
            flag[ir] = F::from(-2).unwrap();
        }
    }

    let mut n_tot = 0;
    for icomp in 0..un_nod {
        // Find seed.
        let mut ii = 0;
        while (flag[ii] >= F::zero()) || (flag[ii] == F::from(-2).unwrap()) {
            ii += 1;
            if ii >= un_nod {
                // Sanity check, if this happens, the graph is corrupted.
                return Err("graph is corrupted".to_string());
            }
        }

        flag[ii] = F::from(icomp).unwrap();
        pos[0] = I::from(ii).unwrap();
        let mut n_pos0 = 0;
        // n_pos_new = n_pos = 1;
        let mut n_pos = 1;
        let mut n_pos_new = n_pos;

        for _ii in 0..un_nod {
            let mut n_new = 0;
            for ir in n_pos0..n_pos {
                let pos_ir = pos[ir].to_usize().unwrap();
                let start = a_p[pos_ir].to_usize().unwrap();
                let end = a_p[pos_ir + 1].to_usize().unwrap();

                for ic in start..end {
                    let aj_ic = a_j[ic].to_usize().unwrap();

                    if flag[aj_ic] == F::from(-1).unwrap() {
                        flag[aj_ic] = F::from(icomp).unwrap();
                        pos[n_pos_new] = a_j[ic];
                        n_pos_new += 1;
                        n_new += 1;
                    }
                }
            }
            n_pos0 = n_pos;
            n_pos = n_pos_new;
            if n_new == 0 {
                break;
            }
        }
        n_tot += n_pos;

        if n_tot == n_stop {
            n_comp = I::from(icomp + 1).unwrap();
            break;
        }
    }

    Ok(n_comp)
}
