use crate::scalar::Scalar;

// pub trait BinaryOp<T: Scalar> {
//     fn binary_op(a: T, b: T) -> T;
// }

pub type Binop<T: Scalar> = fn(T, T) -> T;
