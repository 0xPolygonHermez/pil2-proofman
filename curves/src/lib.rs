mod curve;
mod ecgfp5;
mod ecmasfp5;
mod goldilocks_quintic_extension;

use curve::EllipticCurve;
use goldilocks_quintic_extension::{SquaringFp5, GoldilocksQuinticExtension};

pub use ecgfp5::EcGFp5;
pub use ecmasfp5::EcMasFp5;
