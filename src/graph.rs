//  Author: Jake Vanderplas <vanderplas@astro.washington.edu>

use std::cmp::Ordering;

use crate::csr::CSR;
use crate::traits::{Integer, Scalar};
use anyhow::{format_err, Result};

/// For directed graphs, the type of connection to use.
#[derive(Clone, Copy, PartialEq)]
pub enum Connection {
    /// A directed graph is weakly connected if replacing all of its
    /// directed edges with undirected edges produces a connected
    /// (undirected) graph.
    Weak,
    /// Nodes `i` and `j` are strongly connected if a path exists both
    /// from `i` to `j` and from `j` to `i`.
    Strong,
}

/// Analyze the connected components of a sparse graph
///
/// If directed is `true` (default), then operate on a directed graph: only
/// move from point `i` to point `j` along paths `csgraph[i, j]`.
/// If `false`, then find the shortest path on an undirected graph: the
/// algorithm can progress from point `i` to `j` along `csgraph[i, j]` or
/// `csgraph[j, i]`.
///
/// If `directed` == `false`, the `connection` argument is not referenced.
///
/// Returns the number of connected components and a length-N vector of
/// labels of the connected components.
///
/// # References
///
/// - [1] D. J. Pearce, "An Improved Algorithm for Finding the Strongly
///   Connected Components of a Directed Graph", Technical Report, 2005
///
/// # Example
/// ```
/// use sparsetools::csr::CSR;
/// use sparsetools::graph::{connected_components, Connection};
/// use std::iter::zip;
///
/// fn main() {
///     let graph = CSR::<usize, usize>::from_dense(&vec![
///         vec![0, 1, 1, 0, 0],
///         vec![0, 0, 1, 0, 0],
///         vec![0, 0, 0, 0, 0],
///         vec![0, 0, 0, 0, 1],
///         vec![0, 0, 0, 0, 0]
///     ]);
///     println!("{}", graph.to_table());
///     let (n_components, labels) = connected_components(&graph, false, Connection::Weak).unwrap();
///     println!("{:?}", labels);
///     assert_eq!(n_components, 2);
///     zip(labels, [0, 0, 0, 1, 1]).for_each(|(a, e)| assert_eq!(a, e));
/// }
/// ```
pub fn connected_components<I: Integer, T: Scalar>(
    csgraph: &CSR<I, T>,
    mut directed: bool,
    connection: Connection,
) -> Result<(usize, Vec<usize>)> {
    // weak connections <=> components of undirected graph
    if connection == Connection::Weak {
        directed = false;
    }

    validate_graph(csgraph, directed)?;

    let n = csgraph.cols();

    let (n_components, labels) = if directed {
        connected_components_directed(n, csgraph.colidx(), csgraph.rowptr())
    } else {
        let csgraph_t = csgraph.t().to_csr();
        connected_components_undirected(
            n,
            csgraph.colidx(),
            csgraph.rowptr(),
            csgraph_t.colidx(),
            csgraph_t.rowptr(),
        )
    };

    Ok((n_components, labels))
}

// The array containing the lowlinks of nodes not yet assigned an SCC. Shares
// memory with the labels array, since they are not used at the same time.
macro_rules! lowlinks {
    ($labels:ident) => {
        $labels
    };
}

// stack_f shares memory with SS, as nodes aren't put on the
// SS stack until after they've been popped from the DFS stack.
macro_rules! stack_f {
    ($ss:ident) => {
        $ss
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Node {
    Label(usize),
    Void,
    End,
}

impl Node {
    fn unwrap(self) -> usize {
        match self {
            Node::Label(l) => l,
            Node::Void => panic!("called `Label::unwrap()` on a `Void` value"),
            Node::End => panic!("called `Label::unwrap()` on a `End` value"),
        }
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Node::Label(l1), Node::Label(l2)) => l1.partial_cmp(l2),
            (Node::Label(_), Node::Void) => Some(Ordering::Greater),
            (Node::Label(_), Node::End) => Some(Ordering::Greater),
            (Node::Void, Node::Label(_)) => Some(Ordering::Less),
            (Node::Void, Node::Void) => Some(Ordering::Equal),
            (Node::Void, Node::End) => Some(Ordering::Greater),
            (Node::End, Node::Label(_)) => Some(Ordering::Less),
            (Node::End, Node::Void) => Some(Ordering::Less),
            (Node::End, Node::End) => Some(Ordering::Equal),
        }
    }
}

/// Uses an iterative version of Tarjan's algorithm to find the
/// strongly connected components of a directed graph represented as a
/// sparse matrix.
///
/// The algorithmic complexity is for a graph with `E` edges and `V`
/// vertices is `O(E + V)`.
/// The storage requirement is `2*V` integer arrays.
///
/// Uses an iterative version of the algorithm described here:
/// http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.102.1707
///
/// For more details of the memory optimisations used see here:
/// http://www.timl.id.au/SCC
fn connected_components_directed<I: Integer>(
    n: usize,
    indices: &[I],
    indptr: &[I],
) -> (usize, Vec<usize>) {
    let mut labels = vec![Node::Void; n];

    // The stack of nodes which have been backtracked and are in the current SCC
    let mut ss = vec![Node::Void; n];
    let mut stack_b = vec![Node::Void; n];
    let mut ss_head = Node::End;

    // The DFS stack. Stored with both forwards and backwards pointers to allow
    // us to move a node up to the top of the stack, as we only need to visit
    // each node once.
    let mut stack_head;

    let mut index = 0;
    // Count SCC labels backwards so as not to clash with lowlinks values.
    let mut label = Node::Label(n - 1);
    for v in 0..n {
        if lowlinks![labels][v] == Node::Void {
            // DFS-stack push
            stack_head = Node::Label(v);
            stack_f![ss][v] = Node::End;
            stack_b[v] = Node::End;
            while stack_head != Node::End {
                let v = stack_head.unwrap();
                if lowlinks![labels][v] == Node::Void {
                    lowlinks![labels][v] = Node::Label(index);
                    index += 1;

                    // Add successor nodes
                    for j in indptr[v].to_usize().unwrap()..indptr[v + 1].to_usize().unwrap() {
                        let w = indices[j].to_usize().unwrap();
                        if lowlinks![labels][w] == Node::Void {
                            // TODO: get_unchecked
                            // DFS-stack push
                            if stack_f![ss][w] != Node::Void {
                                // w is already inside the stack,
                                // so excise it.
                                let f = stack_f![ss][w];
                                let b = stack_b[w];
                                if b != Node::End {
                                    stack_f![ss][b.unwrap()] = f;
                                }
                                if f != Node::End {
                                    stack_b[f.unwrap()] = b;
                                }
                            }

                            stack_f![ss][w] = stack_head;
                            stack_b[w] = Node::End;
                            stack_b[stack_head.unwrap()] = Node::Label(w);
                            stack_head = Node::Label(w);
                        }
                    }
                } else {
                    // DFS-stack pop
                    stack_head = stack_f![ss][v];
                    if stack_head != Node::Void && stack_head != Node::End {
                        stack_b[stack_head.unwrap()] = Node::End;
                    }
                    stack_f![ss][v] = Node::Void;
                    stack_b[v] = Node::Void;

                    let mut root = true;
                    let mut low_v = lowlinks![labels][v];
                    for j in indptr[v].to_usize().unwrap()..indptr[v + 1].to_usize().unwrap() {
                        let low_w = lowlinks![labels][indices[j].to_usize().unwrap()];
                        if low_w < low_v {
                            low_v = low_w;
                            root = false;
                        }
                    }
                    lowlinks![labels][v] = low_v;

                    if root {
                        // Found a root node
                        index -= 1;
                        while ss_head != Node::End
                            && lowlinks![labels][v] <= lowlinks![labels][ss_head.unwrap()]
                        {
                            let w = ss_head.unwrap();
                            // w = pop(S)
                            ss_head = ss[w];
                            ss[w] = Node::Void;

                            labels[w] = label;
                            // rindex[w] = c
                            index -= 1;
                        }
                        labels[v] = label;
                        // rindex[v] = c
                        label = match label {
                            Node::Label(l) => {
                                if l == 0 {
                                    Node::Void
                                } else {
                                    Node::Label(l - 1)
                                }
                            }
                            Node::Void => Node::End,
                            Node::End => {
                                unreachable!("label with End node mark");
                            }
                        }
                        // label -= 1; // c = c - 1
                    } else {
                        ss[v] = ss_head;
                        // push(S, v)
                        ss_head = Node::Label(v);
                    }
                }
            }
        }
    }

    // labels count down from N-1 to zero. Modify them so they
    // count upward from 0
    (
        match label {
            Node::Label(l) => (n - 1) - l,
            _ => n, // (n - 1) - (-1),
        },
        labels.iter().map(|l| (n - 1) - l.unwrap()).collect(),
    )
}

// Share memory for the stack and labels, since labels are only
// applied once a node has been popped from the stack.
macro_rules! ss {
    ($labels:ident) => {
        $labels
    };
}

fn connected_components_undirected<I: Integer>(
    n: usize,
    indices1: &[I],
    indptr1: &[I],
    indices2: &[I],
    indptr2: &[I],
) -> (usize, Vec<usize>) {
    let mut labels = vec![Node::Void; n];
    let mut label = 0;

    let mut ss_head;
    for v in 0..n {
        if labels[v] == Node::Void {
            // ss.push(v)
            ss_head = Node::Label(v);
            ss![labels][v] = Node::End;

            while ss_head != Node::End {
                // v = ss.pop()
                let v = ss_head.unwrap();
                ss_head = ss![labels][v];

                labels[v] = Node::Label(label);

                // Push children onto the stack if they haven't been
                // seen at all yet.
                for j in indptr1[v].to_usize().unwrap()..indptr1[v + 1].to_usize().unwrap() {
                    let w = indices1[j].to_usize().unwrap();
                    if ss![labels][w] == Node::Void {
                        ss![labels][w] = ss_head;
                        ss_head = Node::Label(w);
                    }
                }
                for j in indptr2[v].to_usize().unwrap()..indptr2[v + 1].to_usize().unwrap() {
                    let w = indices2[j].to_usize().unwrap();
                    if ss![labels][w] == Node::Void {
                        ss![labels][w] = ss_head;
                        ss_head = Node::Label(w);
                    }
                }
            }
            label += 1;
        }
    }

    (label, labels.iter().map(|l| l.unwrap()).collect())
}

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
///
/// # Example
/// ```
/// use sparsetools::csr::CSR;
/// use sparsetools::graph::depth_first_order;
/// use std::iter::zip;
///
/// fn main() {
///     let graph = CSR::<usize, usize>::from_dense(&vec![
///         vec![0, 1, 2, 0],
///         vec![0, 0, 0, 1],
///         vec![2, 0, 0, 3],
///         vec![0, 0, 0, 0]
///     ]);
///     println!("{}", graph.to_table());
///     let (nodes, predecessors) = depth_first_order(&graph, 0, true).unwrap();
///     println!("{:?}", nodes);
///     println!("{:?}", predecessors);
///     zip(nodes, [0, 1, 3, 2]).for_each(|(a, e)| assert_eq!(a, e));
///     zip(predecessors, [None, Some(0), Some(0), Some(1)]).for_each(|(a, e)| assert_eq!(a, e));
/// }
/// ```
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

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::csr::CSR;
    use crate::graph::{connected_components, Connection};

    #[test]
    fn test_weak_connections() {
        let x_de = vec![vec![0, 1, 0], vec![0, 0, 0], vec![0, 0, 0]];

        let x_sp = CSR::<usize, usize>::from_dense(&x_de);

        let (n_components, labels) = connected_components(&x_sp, true, Connection::Weak).unwrap();

        assert_eq!(n_components, 2);
        zip(labels, [0, 0, 1]).for_each(|(a, e)| assert_eq!(a, e));
    }

    #[test]
    fn test_strong_connections() {
        let x1_de = vec![vec![0, 1, 0], vec![0, 0, 0], vec![0, 0, 0]];
        // X2de = X1de + X1de.T

        let x1_sp = CSR::<usize, usize>::from_dense(&x1_de);
        let x2_sp = &x1_sp + &x1_sp.t().to_csr();

        // println!("{}", x1_sp.to_table());
        // println!("{}", x2_sp.to_table());

        let (n_components, mut labels) =
            connected_components(&x1_sp, true, Connection::Strong).unwrap();

        assert_eq!(n_components, 3);
        labels.sort();
        zip(labels, [0, 1, 2]).for_each(|(a, e)| assert_eq!(a, e));
        // zip(labels, [0, 1, 2]).for_each(|(a, e)| assert_eq!(a, e));

        let (n_components, mut labels) =
            connected_components(&x2_sp, true, Connection::Strong).unwrap();

        assert_eq!(n_components, 2);
        labels.sort();
        zip(labels, [0, 0, 1]).for_each(|(a, e)| assert_eq!(a, e));
        // zip(labels, [0, 0, 1]).for_each(|(a, e)| assert_eq!(a, e));
    }

    #[test]
    fn test_strong_connections2() {
        let x = vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 0, 1, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
            vec![0, 0, 1, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 1, 0],
        ];

        let x_sp = CSR::<usize, usize>::from_dense(&x);

        let (n_components, mut labels) =
            connected_components(&x_sp, true, Connection::Strong).unwrap();

        assert_eq!(n_components, 5);
        labels.sort();
        zip(labels, [0, 1, 2, 2, 3, 4]).for_each(|(a, e)| assert_eq!(a, e));
    }

    #[test]
    fn test_weak_connections2() {
        let x = vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
            vec![0, 0, 1, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 1, 0],
        ];

        let x_sp = CSR::<usize, usize>::from_dense(&x);

        let (n_components, mut labels) =
            connected_components(&x_sp, true, Connection::Weak).unwrap();
        assert_eq!(n_components, 2);
        labels.sort();
        zip(labels, [0, 0, 1, 1, 1, 1]).for_each(|(a, e)| assert_eq!(a, e));
    }
}
