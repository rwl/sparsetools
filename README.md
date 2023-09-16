# sparsetools

2-D sparse matrix crate for numeric data.

The operations supported by SciPy sparse matrices are implemented using
functions written in C++/Cython. This crate contains translations of those
functions to Rust. Sparse matrix types with methods that wrap the functions
are provided in submodules:

- CSR (Compressed Sparse Row format)
- CSC (Compressed Sparse Column format)
- Coo (Coordinate format, aka IJV or triplet format)
- DoK (Dictionary of Keys, backed by `HashMap`)

Some compressed sparse graph routines from SciPy are also included.

## License

The source code is distributed under the same BSD 3-clause license ([LICENSE](LICENSE) or
https://opensource.org/license/bsd-3-clause/) as SciPy.