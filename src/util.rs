use crate::traits::Scalar;

pub type Binop<T: Scalar, T2> = fn(T, T) -> T2;
