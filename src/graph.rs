//  Author: Jake Vanderplas <vanderplas@astro.washington.edu>

use crate::{
    csr::CSR,
    traits::{Integer, Scalar},
};
use anyhow::{format_err, Result};

/// Determine connected components of a compressed sparse graph.
///
/// Note: Output array flag must be preallocated
pub fn cs_graph_components<I: Integer, T: Scalar, F: Integer + num_traits::Signed>(
    n_nod: usize,
    a_p: &[I],
    a_j: &[I],
    flag: &mut [F],
) -> Result<I> {
    // pos is a work array: list of nodes (rows) to process.
    // std::vector<I> pos(n_nod,01);
    let mut pos = vec![I::one(); n_nod];
    let mut n_comp = I::zero();

    let mut n_stop = n_nod;
    for ir in 0..n_nod {
        flag[ir] = F::from(-1).unwrap();
        if a_p[ir + 1] == a_p[ir] {
            n_stop -= 1;
            flag[ir] = F::from(-2).unwrap();
        }
    }

    let mut n_tot = 0;
    for icomp in 0..n_nod {
        // Find seed.
        let mut ii = 0;
        while (flag[ii] >= F::zero()) || (flag[ii] == F::from(-2).unwrap()) {
            ii += 1;
            if ii >= n_nod {
                // Sanity check, if this happens, the graph is corrupted.
                return Err(format_err!("graph is corrupted"));
            }
        }

        flag[ii] = F::from(icomp).unwrap();
        pos[0] = I::from(ii).unwrap();
        let mut n_pos0 = 0;
        // n_pos_new = n_pos = 1;
        let mut n_pos = 1;
        let mut n_pos_new = n_pos;

        for _ii in 0..n_nod {
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

/// Depth-first ordering starting with specified node.
///
/// If `directed=true` (default), then operate on a directed graph: only
/// move from point `i` to point `j` along paths `csgraph[i, j]`.
/// If `false`, then find the shortest path on an undirected graph: the
/// algorithm can progress from point `i` to `j` along `csgraph[i, j]` or
/// `csgraph[j, i]`.
///
/// Returns the depth-first list of nodes, starting with specified node. The
/// length of node_array is the number of nodes reachable from the
/// specified node.
///
/// Also returns the length-`N` list of predecessors of each node in a depth-first
/// tree. If node `i` is in the tree, then its parent is given by
/// `predecessors[i]`. If node `i` is not in the tree (and for the parent
/// node) then `predecessors[i] = None`.
pub fn depth_first_order<I: Integer, T: Scalar>(
    csgraph: &CSR<I, T>,
    i_start: usize,
    directed: bool,
) -> Result<(Vec<usize>, Vec<Option<usize>>)> {
    validate_graph(csgraph, directed)?;
    let n = csgraph.cols();

    let mut node_list = vec![0; n];
    let mut predecessors = vec![None; n];

    let length = if directed {
        depth_first_directed(
            i_start,
            csgraph.colidx(),
            csgraph.rowptr(),
            &mut node_list,
            &mut predecessors,
        )
    } else {
        let csgraph_t = csgraph.t().to_csr();
        depth_first_undirected(
            i_start,
            csgraph.colidx(),
            csgraph.rowptr(),
            csgraph_t.colidx(),
            csgraph_t.rowptr(),
            &mut node_list,
            &mut predecessors,
        )
    };

    Ok((node_list[..length].to_vec(), predecessors))
}

fn depth_first_directed<I: Integer>(
    head_node: usize,
    indices: &[I],
    indptr: &[I],
    node_list: &mut [usize],
    predecessors: &mut [Option<usize>],
) -> usize {
    // cdef unsigned int i, i_nl_end, cnode, pnode
    let n = node_list.len();
    // cdef int no_children, i_root

    let mut root_list = vec![0; n];
    let mut flag = vec![false; n];

    node_list[0] = head_node;
    root_list[0] = head_node;
    let mut i_root: isize = 0;
    let mut i_nl_end = 1;
    flag[head_node] = true;

    while i_root >= 0 {
        let pnode = root_list[i_root as usize];
        let mut no_children = true;
        for i in indptr[pnode].to_usize().unwrap()..indptr[pnode + 1].to_usize().unwrap() {
            let cnode = indices[i].to_usize().unwrap();
            if flag[cnode] {
                continue;
            } else {
                i_root += 1;
                root_list[i_root as usize] = cnode;
                node_list[i_nl_end] = cnode;
                predecessors[cnode] = Some(pnode);
                flag[cnode] = true;
                i_nl_end += 1;
                no_children = false;
                break;
            }
        }

        if i_nl_end == n {
            break;
        }

        if no_children {
            i_root -= 1;
        }
    }
    i_nl_end
}

fn depth_first_undirected<I: Integer>(
    head_node: usize,
    indices1: &[I],
    indptr1: &[I],
    indices2: &[I],
    indptr2: &[I],
    node_list: &mut [usize],
    predecessors: &mut [Option<usize>],
) -> usize {
    // cdef unsigned int i, i_nl_end, cnode, pnode
    let n = node_list.len();
    // cdef int no_children, i_root

    let mut root_list = vec![0; n];
    let mut flag = vec![false; n];

    node_list[0] = head_node;
    root_list[0] = head_node;
    let mut i_root: isize = 0;
    let mut i_nl_end = 1;
    flag[head_node] = true;

    while i_root >= 0 {
        let pnode = root_list[i_root as usize];
        let mut no_children = true;

        for i in indptr1[pnode].to_usize().unwrap()..indptr1[pnode + 1].to_usize().unwrap() {
            let cnode = indices1[i].to_usize().unwrap();
            if flag[cnode] {
                continue;
            } else {
                i_root += 1;
                root_list[i_root as usize] = cnode;
                node_list[i_nl_end] = cnode;
                predecessors[cnode] = Some(pnode);
                flag[cnode] = true;
                i_nl_end += 1;
                no_children = false;
                break;
            }
        }

        if no_children {
            for i in indptr2[pnode].to_usize().unwrap()..indptr2[pnode + 1].to_usize().unwrap() {
                let cnode = indices2[i].to_usize().unwrap();
                if flag[cnode] {
                    continue;
                } else {
                    i_root += 1;
                    root_list[i_root as usize] = cnode;
                    node_list[i_nl_end] = cnode;
                    predecessors[cnode] = Some(pnode);
                    flag[cnode] = true;
                    i_nl_end += 1;
                    no_children = false;
                    break;
                }
            }
        }

        if i_nl_end == n {
            break;
        }

        if no_children {
            i_root -= 1
        }
    }
    i_nl_end
}

fn validate_graph<I: Integer, T: Scalar>(csgraph: &CSR<I, T>, _directed: bool) -> Result<()> {
    if csgraph.rows() != csgraph.cols() {
        return Err(format_err!("compressed-sparse graph must be shape (N, N)"));
    }
    Ok(())
}
