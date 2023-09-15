use crate::traits::{Integer, Scalar};
use pretty_dtoa::FmtFloatConfig;

pub(super) const DEFAULT_FLOAT_CONFIG: FmtFloatConfig = FmtFloatConfig::default()
    .add_point_zero(false)
    .max_significant_digits(6);

/// y += a*x
pub(super) fn axpy<I: Integer, T: Scalar>(n: I, a: T, x: &[T], y: &mut [T]) {
    for i in 0..n.to_usize().unwrap() {
        y[i] += a * x[i];
    }
}
