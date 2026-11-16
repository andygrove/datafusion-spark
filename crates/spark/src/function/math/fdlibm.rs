// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Logarithms that return the same bits as Java's `StrictMath`.
//!
//! Spark evaluates `ln`, `log`, `log10`, `log2` and `log1p` with
//! `java.lang.StrictMath`, which the Java specification requires to return
//! exactly the results of fdlibm 5.3. Rust's `f64::ln` calls the platform's
//! libm, which differs from fdlibm in the last bit for a few percent of inputs,
//! and differently on each platform. The `libm` crate follows musl, which
//! simplified fdlibm's final steps and so also rounds differently.
//!
//! These functions are a direct translation of fdlibm 5.3's `e_log.c`,
//! `e_log10.c` and `s_log1p.c` from C to Rust, keeping its order of operations.
//! They match `StrictMath` bit for bit on 2 million random inputs, on JDK 17
//! and JDK 21. Special values (0, negative numbers, NaN and infinities) give
//! the IEEE 754 results. The callers map the inputs that Spark sends to NULL
//! before calling these functions.
//!
//! The original fdlibm source carries this notice, which must be preserved:
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```

// fdlibm's constants, as the bit patterns given in its comments.
// `log1p`'s Lp1-Lp7 have the same values as `log`'s Lg1-Lg7.
const LN2_HI: f64 = f64::from_bits(0x3fe62e42_fee00000);
const LN2_LO: f64 = f64::from_bits(0x3dea39ef_35793c76);
const TWO54: f64 = f64::from_bits(0x43500000_00000000);
const LG1: f64 = f64::from_bits(0x3fe55555_55555593);
const LG2: f64 = f64::from_bits(0x3fd99999_9997fa04);
const LG3: f64 = f64::from_bits(0x3fd24924_94229359);
const LG4: f64 = f64::from_bits(0x3fcc71c5_1d8e78af);
const LG5: f64 = f64::from_bits(0x3fc74664_96cb03de);
const LG6: f64 = f64::from_bits(0x3fc39a09_d078c69f);
const LG7: f64 = f64::from_bits(0x3fc2f112_df3e5244);

/// The high 32 bits of `x`, as fdlibm's `__HI`.
fn hi(x: f64) -> i32 {
    (x.to_bits() >> 32) as i32
}
/// The low 32 bits of `x`, as fdlibm's `__LO`.
fn lo(x: f64) -> u32 {
    x.to_bits() as u32
}
/// `x` with its high 32 bits replaced, as an assignment to fdlibm's `__HI`.
fn with_hi(x: f64, hi: i32) -> f64 {
    f64::from_bits(((hi as u32 as u64) << 32) | (x.to_bits() & 0xffff_ffff))
}

/// `StrictMath.log`: the natural logarithm (fdlibm `__ieee754_log`).
pub(crate) fn log(mut x: f64) -> f64 {
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        if ((hx & 0x7fff_ffff) as u32 | lx) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = hi(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x10_0000;
    x = with_hi(x, hx | (i ^ 0x3ff0_0000));
    k += i >> 20;
    let f = x - 1.0;
    let dk = k as f64;
    if (0x000f_ffff & (2 + hx)) < 3 {
        if f == 0.0 {
            return if k == 0 {
                0.0
            } else {
                dk * LN2_HI + dk * LN2_LO
            };
        }
        let r = f * f * (0.5 - f64::from_bits(0x3fd55555_55555555) * f);
        return if k == 0 {
            f - r
        } else {
            dk * LN2_HI - ((r - dk * LN2_LO) - f)
        };
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let mut i = hx - 0x6147a;
    let w = z * z;
    let j = 0x6b851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

const IVLN10: f64 = f64::from_bits(0x3fdbcb7b_1526e50e);
const LOG10_2HI: f64 = f64::from_bits(0x3fd34413_509f6000);
const LOG10_2LO: f64 = f64::from_bits(0x3d59fef3_11f12b36);

/// `StrictMath.log10`: the base 10 logarithm (fdlibm `__ieee754_log10`).
pub(crate) fn log10(mut x: f64) -> f64 {
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        if ((hx & 0x7fff_ffff) as u32 | lx) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = hi(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    let i = ((k as u32) & 0x8000_0000) >> 31;
    hx = (hx & 0x000f_ffff) | ((0x3ff - i as i32) << 20);
    let y = (k + i as i32) as f64;
    x = with_hi(x, hx);
    let z = y * LOG10_2LO + IVLN10 * log(x);
    z + y * LOG10_2HI
}

/// `StrictMath.log1p`: `ln(1 + x)`, accurate near 0 (fdlibm `log1p`).
pub(crate) fn log1p(x: f64) -> f64 {
    let hx = hi(x);
    let ax = hx & 0x7fff_ffff;
    let mut k: i32 = 1;
    let mut f = 0.0;
    let mut hu: i32 = 0;
    let mut c = 0.0;
    if hx < 0x3FDA_827A {
        if ax >= 0x3ff0_0000 {
            return if x == -1.0 {
                f64::NEG_INFINITY
            } else {
                f64::NAN
            };
        }
        if ax < 0x3e20_0000 {
            return if TWO54 + x > 0.0 && ax < 0x3c90_0000 {
                x
            } else {
                x - x * x * 0.5
            };
        }
        if hx > 0 || hx <= 0xbfd2_bec3_u32 as i32 {
            k = 0;
            f = x;
            hu = 1;
        }
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    if k != 0 {
        let mut u;
        if hx < 0x4340_0000 {
            u = 1.0 + x;
            hu = hi(u);
            k = (hu >> 20) - 1023;
            c = if k > 0 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            u = x;
            hu = hi(u);
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000f_ffff;
        if hu < 0x6a09e {
            u = with_hi(u, hu | 0x3ff0_0000);
        } else {
            k += 1;
            u = with_hi(u, hu | 0x3fe0_0000);
            hu = (0x0010_0000 - hu) >> 2;
        }
        f = u - 1.0;
    }
    let hfsq = 0.5 * f * f;
    let dk = k as f64;
    if hu == 0 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            c += dk * LN2_LO;
            return dk * LN2_HI + c;
        }
        let r = hfsq * (1.0 - f64::from_bits(0x3fe55555_55555555) * f);
        return if k == 0 {
            f - r
        } else {
            dk * LN2_HI - ((r - (dk * LN2_LO + c)) - f)
        };
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let r =
        z * (LG1 + z * (LG2 + z * (LG3 + z * (LG4 + z * (LG5 + z * (LG6 + z * LG7))))));
    if k == 0 {
        f - (hfsq - s * (hfsq + r))
    } else {
        dk * LN2_HI - ((hfsq - (s * (hfsq + r) + (dk * LN2_LO + c))) - f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inputs with the raw bits of `StrictMath.log`, `log10` and `log1p` on
    /// JDK 17. The first two are inputs where Rust's `f64::ln` is 1 ulp off.
    const STRICT_MATH: &[(f64, i64, i64, i64)] = &[
        (
            0.670311110015402,
            -4622494417381967792,
            -4627944964877657846,
            4602796001956754528,
        ),
        (
            0.801786884658205,
            -4626244807238519389,
            -4631794334810663820,
            4603478468290144808,
        ),
        (
            81.0,
            4616633728321258251,
            4611273871592510973,
            4616647543217364325,
        ),
        (
            3.0,
            4607626529066517258,
            4602266672337769981,
            4608922133940812271,
        ),
        (
            1000.0,
            4619463459452485536,
            4613937818241073152,
            4619464584789817444,
        ),
        (
            2.0,
            4604418534313441775,
            4599094494223104511,
            4607626529066517258,
        ),
        (4.9e-324, -4573612656913714749, -4578976757573532812, 1),
        (
            f64::MAX,
            4649454530587146735,
            4644130490496809471,
            4649454530587146735,
        ),
        (
            0.5658035575800715,
            -4620067142282017771,
            -4625292858330671666,
            4601749260853035305,
        ),
        (
            806916.0,
            4623846286214522504,
            4618336515890487656,
            4623846286912178276,
        ),
        (
            0.4111496273080162,
            -4617191235959499470,
            -4622746857327404785,
            4599875863624971816,
        ),
        (
            1e-20,
            -4591694018691028389,
            -4597049319638433792,
            4307583784117748259,
        ),
        (
            1e-300,
            -4574084695234936907,
            -4579386764849840128,
            118622047889322841,
        ),
    ];

    #[test]
    fn matches_strict_math() {
        for &(x, ln, lg10, ln1p) in STRICT_MATH {
            assert_eq!(log(x).to_bits() as i64, ln, "log({x})");
            assert_eq!(log10(x).to_bits() as i64, lg10, "log10({x})");
            assert_eq!(log1p(x).to_bits() as i64, ln1p, "log1p({x})");
        }
    }

    #[test]
    fn log1p_negative_inputs() {
        assert_eq!(log1p(-0.5), -std::f64::consts::LN_2);
        assert_eq!(log1p(-1e-10), -1.00000000005e-10);
    }

    #[test]
    fn special_values() {
        for f in [log, log10] {
            assert_eq!(f(0.0), f64::NEG_INFINITY);
            assert_eq!(f(-0.0), f64::NEG_INFINITY);
            assert!(f(-1.0).is_nan());
            assert!(f(f64::NAN).is_nan());
            assert_eq!(f(f64::INFINITY), f64::INFINITY);
            assert_eq!(f(1.0), 0.0);
        }
        assert_eq!(log1p(-1.0), f64::NEG_INFINITY);
        assert!(log1p(-2.0).is_nan());
        assert!(log1p(f64::NAN).is_nan());
        assert_eq!(log1p(f64::INFINITY), f64::INFINITY);
        assert_eq!(log1p(0.0), 0.0);
        assert_eq!(log1p(-0.0).to_bits(), (-0.0f64).to_bits());
    }
}
