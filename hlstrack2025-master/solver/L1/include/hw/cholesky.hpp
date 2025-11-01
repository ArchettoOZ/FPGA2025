/*
 * Copyright 2021 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file cholesky.hpp
 * @brief This file contains cholesky functions
 *   - cholesky                 : Entry point function
 *   - choleskyTop             : Top level function that selects implementation architecture and internal types based
 * on a traits class.
 *   - choleskyBasic           : Basic implementation requiring lower resource
 *   - choleskyAlt             : Lower latency architecture requiring more resources
 *   - choleskyAlt2            : Further improved latency architecture requiring higher resource
 */

#ifndef _XF_SOLVER_CHOLESKY_HPP_
#define _XF_SOLVER_CHOLESKY_HPP_

#include "ap_fixed.h"
#include "hls_x_complex.h"
#include <complex>
#include "ap_fixed_sqrt.hpp"
#include "utils/std_complex_utils.h"
#include "utils/x_matrix_utils.hpp"
#include "hls_stream.h"

namespace xf {
namespace solver {

// ===================================================================================================================
// Default traits struct defining the internal variable types for the cholesky function
template <bool LowerTriangularL, int RowsColsA, typename InputType, typename OutputType>
struct choleskyTraits {
    typedef InputType PROD_T;
    typedef InputType ACCUM_T;
    typedef InputType ADD_T;
    typedef InputType DIAG_T;
    typedef InputType RECIP_DIAG_T;
    typedef InputType OFF_DIAG_T;
    typedef OutputType L_OUTPUT_T;
    static const int ARCH =
        0; // Select implementation: 0=Basic, 1=Lower latency architecture, 2=Further improved latency architecture
    static const int INNER_II = 1; // Specify the pipelining target for the inner loop
    static const int UNROLL_FACTOR =
        1; // Specify the inner loop unrolling factor for the choleskyAlt2 architecture(2) to increase throughput
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2); // Dimension to unroll matrix
    static const int ARCH2_ZERO_LOOP =
        true; // Additional implementation "switch" for the choleskyAlt2 architecture (2).
};

// Specialization for complex
template <bool LowerTriangularL, int RowsColsA, typename InputBaseType, typename OutputBaseType>
struct choleskyTraits<LowerTriangularL, RowsColsA, hls::x_complex<InputBaseType>, hls::x_complex<OutputBaseType> > {
    typedef hls::x_complex<InputBaseType> PROD_T;
    typedef hls::x_complex<InputBaseType> ACCUM_T;
    typedef hls::x_complex<InputBaseType> ADD_T;
    typedef hls::x_complex<InputBaseType> DIAG_T;
    typedef InputBaseType RECIP_DIAG_T;
    typedef hls::x_complex<InputBaseType> OFF_DIAG_T;
    typedef hls::x_complex<OutputBaseType> L_OUTPUT_T;
    static const int ARCH = 0;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
};

// Specialization for std complex
template <bool LowerTriangularL, int RowsColsA, typename InputBaseType, typename OutputBaseType>
struct choleskyTraits<LowerTriangularL, RowsColsA, std::complex<InputBaseType>, std::complex<OutputBaseType> > {
    typedef std::complex<InputBaseType> PROD_T;
    typedef std::complex<InputBaseType> ACCUM_T;
    typedef std::complex<InputBaseType> ADD_T;
    typedef std::complex<InputBaseType> DIAG_T;
    typedef InputBaseType RECIP_DIAG_T;
    typedef std::complex<InputBaseType> OFF_DIAG_T;
    typedef std::complex<OutputBaseType> L_OUTPUT_T;
    static const int ARCH = 0;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
};

// Specialization for ap_fixed
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL, RowsColsA, ap_fixed<W1, I1, Q1, O1, N1>, ap_fixed<W2, I2, Q2, O2, N2> > {
    typedef ap_fixed<W1 + W1, I1 + I1, AP_RND_CONV, AP_SAT, 0> PROD_T;
    typedef ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value,
                     (I1 + I1) + BitWidth<RowsColsA>::Value,
                     AP_RND_CONV,
                     AP_SAT,
                     0>
        ACCUM_T;
    typedef ap_fixed<W1 + 1, I1 + 1, AP_RND_CONV, AP_SAT, 0> ADD_T;
    // Add small fractional guard (+2 bits) to reduce quantization when narrowing
    typedef ap_fixed<(W1 + 1) * 2 + 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> DIAG_T;     // Takes result of sqrt
    typedef ap_fixed<(W1 + 1) * 2 + 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2 + 2, 2 + (W2 - I2), AP_RND_CONV, AP_SAT, 0> RECIP_DIAG_T;
    typedef ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0>
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = 0;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
};

// Further specialization for hls::complex<ap_fixed>
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL,
                      RowsColsA,
                      hls::x_complex<ap_fixed<W1, I1, Q1, O1, N1> >,
                      hls::x_complex<ap_fixed<W2, I2, Q2, O2, N2> > > {
    typedef hls::x_complex<ap_fixed<W1 + W1, I1 + I1, AP_RND_CONV, AP_SAT, 0> > PROD_T;
    typedef hls::x_complex<ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value,
                                    (I1 + I1) + BitWidth<RowsColsA>::Value,
                                    AP_RND_CONV,
                                    AP_SAT,
                                    0> >
        ACCUM_T;
    typedef hls::x_complex<ap_fixed<W1 + 1, I1 + 1, AP_RND_CONV, AP_SAT, 0> > ADD_T;
    // Add small fractional guard (+2 bits) to reduce quantization when narrowing
    typedef hls::x_complex<ap_fixed<(W1 + 1) * 2 + 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > DIAG_T;     // Takes result of sqrt
    typedef hls::x_complex<ap_fixed<(W1 + 1) * 2 + 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2 + 2, 2 + (W2 - I2), AP_RND_CONV, AP_SAT, 0> RECIP_DIAG_T;
    typedef hls::x_complex<ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0> >
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = 0;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
};

// Further specialization for std::complex<ap_fixed>
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL,
                      RowsColsA,
                      std::complex<ap_fixed<W1, I1, Q1, O1, N1> >,
                      std::complex<ap_fixed<W2, I2, Q2, O2, N2> > > {
    typedef std::complex<ap_fixed<W1 + W1, I1 + I1, AP_RND_CONV, AP_SAT, 0> > PROD_T;
    typedef std::complex<ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value,
                                  (I1 + I1) + BitWidth<RowsColsA>::Value,
                                  AP_RND_CONV,
                                  AP_SAT,
                                  0> >
        ACCUM_T;
    typedef std::complex<ap_fixed<W1 + 1, I1 + 1, AP_RND_CONV, AP_SAT, 0> > ADD_T;
    // Add small fractional guard (+2 bits) to reduce quantization when narrowing
    typedef std::complex<ap_fixed<(W1 + 1) * 2 + 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > DIAG_T;     // Takes result of sqrt
    typedef std::complex<ap_fixed<(W1 + 1) * 2 + 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2 + 2, 2 + (W2 - I2), AP_RND_CONV, AP_SAT, 0> RECIP_DIAG_T;
    typedef std::complex<ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0> >
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = 0;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
};

template <int W, int I, ap_q_mode Q, ap_o_mode O, int N, typename T_OUT>
int cholesky_sqrt_op(ap_fixed<W, I, Q, O, N> a, T_OUT& b) {
Function_cholesky_sqrt_op_fixed:;
    const ap_fixed<W, I, Q, O, N> ZERO = 0;
    const ap_fixed<W, I, Q, O, N> EPSILON = ap_fixed<W, I, Q, O, N>(1e-15);
    if (a < ZERO) {
        b = ZERO;
        return (1);
    }
    if (a < EPSILON) {
        b = ZERO;
        return (0);
    }
    ap_fixed<W, I, Q, O, N> tmp = fixed_sqrt_fast<W, I, 6, 12>(a);
    b = tmp;
    return (0);
}

template <int W, int I, ap_q_mode Q, ap_o_mode O, int N, typename T_OUT>
int cholesky_sqrt_op(hls::x_complex<ap_fixed<W, I, Q, O, N> > din, hls::x_complex<T_OUT>& dout) {
Function_cholesky_sqrt_op_fixed_complex:;
    const ap_fixed<W, I, Q, O, N> ZERO = 0;
    const ap_fixed<W, I, Q, O, N> EPSILON = ap_fixed<W, I, Q, O, N>(1e-15);
    ap_fixed<W, I, Q, O, N> a = din.real();
    dout.imag(ZERO);
    if (a < ZERO) {
        dout.real(ZERO);
        return (1);
    }
    if (a < EPSILON) {
        dout.real(ZERO);
        return (0);
    }
    ap_fixed<W, I, Q, O, N> tmp = fixed_sqrt_fast<W, I, 6, 12>(a);
    dout.real(tmp);
    return (0);
}

template <int W, int I, ap_q_mode Q, ap_o_mode O, int N, typename T_OUT>
int cholesky_sqrt_op(std::complex<ap_fixed<W, I, Q, O, N> > din, std::complex<T_OUT>& dout) {
Function_cholesky_sqrt_op_fixed_std_complex:;
    const ap_fixed<W, I, Q, O, N> ZERO = 0;
    const ap_fixed<W, I, Q, O, N> EPSILON = ap_fixed<W, I, Q, O, N>(1e-15);
    ap_fixed<W, I, Q, O, N> a = din.real();
    dout.imag(ZERO);
    if (a < ZERO) {
        dout.real(ZERO);
        return (1);
    }
    if (a < EPSILON) {
        dout.real(ZERO);
        return (0);
    }
    ap_fixed<W, I, Q, O, N> tmp = fixed_sqrt_fast<W, I, 6, 12>(a);
    dout.real(tmp);
    return (0);
}
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(T_IN a, T_OUT& b) {
Function_cholesky_sqrt_op_real:;
    const T_IN ZERO = 0;
    if (a < ZERO) {
        b = ZERO;
        return (1);
    }
    b = x_sqrt(a);
    return (0);
}
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(hls::x_complex<T_IN> din, hls::x_complex<T_OUT>& dout) {
Function_cholesky_sqrt_op_complex:;
    const T_IN ZERO = 0;
    T_IN a = din.real();
    dout.imag(ZERO);

    if (a < ZERO) {
        dout.real(ZERO);
        return (1);
    }

    dout.real(x_sqrt(a));
    return (0);
}
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(std::complex<T_IN> din, std::complex<T_OUT>& dout) {
Function_cholesky_sqrt_op_complex:;
    const T_IN ZERO = 0;
    T_IN a = din.real();
    dout.imag(ZERO);

    if (a < ZERO) {
        dout.real(ZERO);
        return (1);
    }

    dout.real(x_sqrt(a));
    return (0);
}

// Reciprocal square root.
template <typename InputType, typename OutputType>
void cholesky_rsqrt(InputType x, OutputType& res) {
Function_cholesky_rsqrt_default:;
    res = x_rsqrt(x);
}
template <int W1, int I1, ap_q_mode Q1, ap_o_mode O1, int N1, int W2, int I2, ap_q_mode Q2, ap_o_mode O2, int N2>
void cholesky_rsqrt(ap_fixed<W1, I1, Q1, O1, N1> x, ap_fixed<W2, I2, Q2, O2, N2>& res) {
Function_cholesky_rsqrt_fixed:;
    ap_fixed<W1, I1, Q1, O1, N1> rsqrt_tmp = fixed_rsqrt_fast<W1, I1, 6, 3>(x);
    res = rsqrt_tmp;
}

// Local multiplier to handle a complex case currently not supported by the hls::x_complex class
// - Complex multiplied by a real of a different type
// - Required for complex fixed point implementations
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(AType A, BType B, CType& C) {
Function_cholesky_prod_sum_mult_real:;
    C = A * B;
}
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(hls::x_complex<AType> A, BType B, hls::x_complex<CType>& C) {
Function_cholesky_prod_sum_mult_complex:;
    C.real(A.real() * B);
    C.imag(A.imag() * B);
}
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(std::complex<AType> A, BType B, std::complex<CType>& C) {
Function_cholesky_prod_sum_mult_complex:;
    C.real(A.real() * B);
    C.imag(A.imag() * B);
}

// Divide by a real scalar (helper)
template <typename AType, typename BType, typename CType>
void cholesky_div_by_real(AType A, BType denom, CType& C) {
Function_cholesky_div_by_real_real:;
    C = A / denom;
}
template <typename AType, typename BType, typename CType>
void cholesky_div_by_real(hls::x_complex<AType> A, BType denom, hls::x_complex<CType>& C) {
Function_cholesky_div_by_real_complex:;
    C.real(A.real() / denom);
    C.imag(A.imag() / denom);
}
template <typename AType, typename BType, typename CType>
void cholesky_div_by_real(std::complex<AType> A, BType denom, std::complex<CType>& C) {
Function_cholesky_div_by_real_std_complex:;
    C.real(A.real() / denom);
    C.imag(A.imag() / denom);
}

// ===================================================================================================================
// choleskyBasic
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyBasic(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    int return_code = 0;

    // Upper-triangular result is obtained from the conjugate-transpose of the lower solution.
    if (LowerTriangularL == false) {
        typedef choleskyTraits<true, RowsColsA, InputType, OutputType> lower_traits_default;
        OutputType L_lower[RowsColsA][RowsColsA];
        int lower_return = choleskyBasic<true, RowsColsA, lower_traits_default, InputType, OutputType>(A, L_lower);
    upper_fill_loop:
        for (int r = 0; r < RowsColsA; ++r) {
#pragma HLS PIPELINE
            for (int c = 0; c < RowsColsA; ++c) {
                if (r <= c) {
                    L[r][c] = hls::x_conj(L_lower[c][r]);
                } else {
                    L[r][c] = 0;
                }
            }
        }
        return lower_return;
    }

    // Use the traits struct to specify the correct type for the intermediate variables. This is really only needed for
    // fixed point.
    typename CholeskyTraits::PROD_T prod;
    typename CholeskyTraits::ACCUM_T A_cast_to_sum;    // A with the same dimensions as sum.
    typename CholeskyTraits::ACCUM_T prod_cast_to_sum; // prod with the same dimensions as sum.

    typename CholeskyTraits::ACCUM_T A_minus_sum;
    typename CholeskyTraits::DIAG_T new_L_diag;         // sqrt(A_minus_sum)
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag; // sum/L
    typename CholeskyTraits::OFF_DIAG_T L_cast_to_new_L_off_diag;
    typename CholeskyTraits::RECIP_DIAG_T new_L_diag_recip;
    typename CholeskyTraits::DIAG_T A_minus_sum_cast_diag; 

    typename CholeskyTraits::L_OUTPUT_T new_L;
    typename CholeskyTraits::OFF_DIAG_T retrieved_L;
    typename CholeskyTraits::OFF_DIAG_T L_internal[RowsColsA][RowsColsA];
    InputType A_local[RowsColsA][RowsColsA];
symm_rows:
    for (int r = 0; r < RowsColsA; ++r) {
#pragma HLS PIPELINE
        for (int c = 0; c < RowsColsA; ++c) {
            A_local[r][c] = A[r][c];
        }
    }

init_internal:
    for (int r = 0; r < RowsColsA; ++r) {
#pragma HLS PIPELINE
        for (int c = 0; c < RowsColsA; ++c) {
            L_internal[r][c] = typename CholeskyTraits::OFF_DIAG_T();
        }
    }

col_loop:
    for (int j = 0; j < RowsColsA; j++) {
        typename CholeskyTraits::ACCUM_T diag_sum = typename CholeskyTraits::ACCUM_T();

    // Calculate the diagonal value for this column
    diag_loop:
        for (int k = 0; k < RowsColsA; k++) {
            if (k < j) {
                if (LowerTriangularL == true) {
                    retrieved_L = L_internal[j][k];
                } else {
                    retrieved_L = L_internal[k][j];
                }
                diag_sum += hls::x_conj(retrieved_L) * retrieved_L;
            }
        }
    A_cast_to_sum = A_local[j][j];

    A_minus_sum = A_cast_to_sum - diag_sum;

        if (cholesky_sqrt_op(A_minus_sum, new_L_diag)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }
    {
        auto Ljj_real = hls::x_real(new_L_diag);
        if (Ljj_real != (decltype(Ljj_real))0) {
            new_L_diag_recip = (typename CholeskyTraits::RECIP_DIAG_T)((typename CholeskyTraits::RECIP_DIAG_T)1) /
                               (typename CholeskyTraits::RECIP_DIAG_T)Ljj_real;
        } else {
            new_L_diag_recip = 0;
        }
    }
    new_L = new_L_diag;

        if (LowerTriangularL == true) {
            L_internal[j][j] = (typename CholeskyTraits::OFF_DIAG_T)new_L_diag;
            L[j][j] = new_L;
        } else {
            L_internal[j][j] = (typename CholeskyTraits::OFF_DIAG_T)new_L_diag;
            L[j][j] = hls::x_conj(new_L);
        }

    // Calculate the off diagonal values for this column
    off_diag_loop:
        for (int i = 0; i < RowsColsA; i++) {
            if (i > j) {
                if (hls::x_real(new_L_diag) == (decltype(hls::x_real(new_L_diag)))0) {
                    if (LowerTriangularL == true) {
                        L[i][j] = 0;
                        L_internal[i][j] = 0;
                    } else {
                        L[j][i] = 0;
                        L_internal[j][i] = 0;
                    }
                    continue;
                }
                
                // Initialize off-diagonal sum for this element
                typename CholeskyTraits::ACCUM_T off_diag_sum;
                if (LowerTriangularL == true) {
                    off_diag_sum = A_local[i][j];
                } else {
                    off_diag_sum = A_local[j][i];
                }

            sum_loop:
                for (int k = 0; k < RowsColsA; k++) {
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
                    if (k < j) {
                        if (LowerTriangularL == true) {
                            prod = L_internal[i][k] * hls::x_conj(L_internal[j][k]);
                        } else {
                            prod = hls::x_conj(L_internal[k][j]) * L_internal[k][i];
                        }

                        prod_cast_to_sum = prod;
                        off_diag_sum -= prod_cast_to_sum;
                    }
                }
                typename CholeskyTraits::ACCUM_T tmp_off_scaled;
                auto Ljj_real_div = hls::x_real(new_L_diag);
                cholesky_div_by_real(off_diag_sum, Ljj_real_div, tmp_off_scaled);
                new_L = (typename CholeskyTraits::L_OUTPUT_T)tmp_off_scaled;
                new_L_off_diag = (typename CholeskyTraits::OFF_DIAG_T)tmp_off_scaled;

                if (LowerTriangularL == true) {
                    L[i][j] = new_L;
                    L_internal[i][j] = new_L_off_diag;
                } else {
                    L[j][i] = new_L;
                    L_internal[j][i] = new_L_off_diag;
                }
            } else if (i < j) {
                if (LowerTriangularL == true) {
                    L[i][j] = 0;
                } else {
                    L[j][i] = 0;
                }
            }
        }
    }
    return (return_code);
}

template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyAlt(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    int return_code = 0;

    if (LowerTriangularL == false) {
        typedef choleskyTraits<true, RowsColsA, InputType, OutputType> lower_traits_default;
        OutputType L_lower[RowsColsA][RowsColsA];
        int lower_return = choleskyAlt<true, RowsColsA, lower_traits_default, InputType, OutputType>(A, L_lower);
    upper_fill_loop_alt:
        for (int r = 0; r < RowsColsA; ++r) {
#pragma HLS PIPELINE
            for (int c = 0; c < RowsColsA; ++c) {
                if (r <= c) {
                    L[r][c] = hls::x_conj(L_lower[c][r]);
                } else {
                    L[r][c] = 0;
                }
            }
        }
        return lower_return;
    }
    typedef ap_fixed<48, 16, AP_RND_CONV, AP_SAT, 0> wide_real_t;
    typedef hls::x_complex<wide_real_t> wide_cplx_t;
    wide_cplx_t A_wide[RowsColsA][RowsColsA];
    wide_cplx_t L_internal_wide[(RowsColsA * RowsColsA - RowsColsA) / 2];
    wide_real_t diag_recip_wide[RowsColsA];
    for (int r = 0; r < RowsColsA; ++r) {
        for (int c = 0; c < RowsColsA; ++c) {
#pragma HLS PIPELINE
            wide_cplx_t tmp;
            tmp.real((wide_real_t)hls::x_real(A[r][c]));
            tmp.imag((wide_real_t)hls::x_imag(A[r][c]));
            A_wide[r][c] = tmp;
        }
    }
row_loop:
    for (int i = 0; i < RowsColsA; i++) {
        int i_sub1 = i - 1;
        int i_off = (i_sub1 > 0) ? ((i_sub1 * i_sub1 - i_sub1) / 2) + i_sub1 : 0;
        wide_cplx_t zero_c = wide_cplx_t((wide_real_t)0, (wide_real_t)0);
        wide_cplx_t square_sum = zero_c;
    col_loop:
        for (int j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT max = 1 + RowsColsA / 2
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
            int j_sub1 = j - 1;
            int j_off = (j_sub1 > 0) ? ((j_sub1 * j_sub1 - j_sub1) / 2) + j_sub1 : 0;
            wide_cplx_t product_sum;
            if (LowerTriangularL) {
                product_sum = A_wide[i][j];
            } else {
                product_sum = A_wide[j][i];
            }
            for (int k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT max = 1 + RowsColsA / 2
#pragma HLS PIPELINE II = 1
                wide_cplx_t Lik = L_internal_wide[i_off + k];
                wide_cplx_t Ljk = L_internal_wide[j_off + k];
                wide_cplx_t prod = Lik * hls::x_conj(Ljk);
                product_sum -= prod;
            }
            wide_real_t diag_r = diag_recip_wide[j];
            wide_cplx_t new_L_off;
            cholesky_prod_sum_mult(product_sum, diag_r, new_L_off);
            L_internal_wide[i_off + j] = new_L_off;
            OutputType out;
            hls::x_real(out) = (decltype(hls::x_real(out)))new_L_off.real();
            hls::x_imag(out) = (decltype(hls::x_imag(out)))new_L_off.imag();

            if (LowerTriangularL) {
                L[i][j] = out;
                L[j][i] = 0;
            } else {
                L[j][i] = out;
                L[i][j] = 0;
            }
            square_sum += hls::x_conj(new_L_off) * new_L_off;
        } 
        wide_cplx_t Aii = A_wide[i][i];
        wide_cplx_t A_minus_sum = Aii - square_sum;
        wide_real_t diag_real = A_minus_sum.real();
        if (diag_real < (wide_real_t)0) {
#ifndef __SYNTHESIS__
            printf("ERROR: negative diag in wide computation at i=%d: %f\n", i, (double)diag_real);
#endif
            return_code = 1;
        }
        ap_fixed<48, 16, AP_RND_CONV, AP_SAT, 0> diag_in = (ap_fixed<48, 16, AP_RND_CONV, AP_SAT, 0>)diag_real;
    ap_fixed<48, 16, AP_RND_CONV, AP_SAT, 0> diag_sqrt_wide = fixed_sqrt_fast<48, 16, 6, 12>(diag_in);
        typename CholeskyTraits::DIAG_T new_L_diag;
        hls::x_real(new_L_diag) = (decltype(hls::x_real(new_L_diag)))diag_sqrt_wide;
        hls::x_imag(new_L_diag) = (decltype(hls::x_imag(new_L_diag)))((ap_fixed<48,16>)0);
        if (diag_sqrt_wide != (ap_fixed<48, 16, AP_RND_CONV, AP_SAT, 0>)0) {
            diag_recip_wide[i] = (wide_real_t)((ap_fixed<48, 16, AP_RND_CONV, AP_SAT, 0>)1) / (wide_real_t)diag_sqrt_wide;
        } else {
            diag_recip_wide[i] = (wide_real_t)0;
        }
        OutputType out_diag;
        hls::x_real(out_diag) = (decltype(hls::x_real(out_diag)))hls::x_real(new_L_diag);
        hls::x_imag(out_diag) = (decltype(hls::x_imag(out_diag)))((ap_fixed<48,16>)0);

        if (LowerTriangularL) {
            L[i][i] = out_diag;
        } else {
            L[i][i] = hls::x_conj(out_diag);
        }
    }

    return (return_code);
}

// ===================================================================================================================
// choleskyAlt2: Further improved latency architecture requiring higher resource
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyAlt2(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    int return_code = 0;

    // To avoid array index calculations every iteration this architecture uses a simple 2D array rather than a
    // optimized/packed triangular matrix.
    OutputType L_internal[RowsColsA][RowsColsA];
    OutputType prod_column_top;
    typename CholeskyTraits::ACCUM_T square_sum_array[RowsColsA];
    typename CholeskyTraits::ACCUM_T A_cast_to_sum;
    typename CholeskyTraits::ADD_T A_minus_sum;
    typename CholeskyTraits::DIAG_T A_minus_sum_cast_diag;
    typename CholeskyTraits::DIAG_T new_L_diag;
    typename CholeskyTraits::RECIP_DIAG_T new_L_diag_recip;
    typename CholeskyTraits::PROD_T prod;
    typename CholeskyTraits::ACCUM_T prod_cast_to_sum;
    typename CholeskyTraits::ACCUM_T product_sum;
    typename CholeskyTraits::ACCUM_T product_sum_array[RowsColsA];
    typename CholeskyTraits::OFF_DIAG_T prod_cast_to_off_diag;
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag;
    typename CholeskyTraits::L_OUTPUT_T new_L;

#pragma HLS ARRAY_PARTITION variable = A cyclic dim = CholeskyTraits::UNROLL_DIM factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = L cyclic dim = CholeskyTraits::UNROLL_DIM factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = L_internal cyclic dim = CholeskyTraits::UNROLL_DIM factor = \
    CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = square_sum_array cyclic dim = 1 factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = product_sum_array cyclic dim = 1 factor = CholeskyTraits::UNROLL_FACTOR

col_loop:
    for (int j = 0; j < RowsColsA; j++) {
        // Diagonal calculation
        A_cast_to_sum = A[j][j];
        if (j == 0) {
            A_minus_sum = A_cast_to_sum;
        } else {
            A_minus_sum = A_cast_to_sum - square_sum_array[j];
        }
        if (cholesky_sqrt_op(A_minus_sum, new_L_diag)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }
        // Round to target format using method specifed by traits defined types.
        new_L = new_L_diag;
        {
            auto Ljj_real = hls::x_real(new_L_diag);
            if (Ljj_real != (decltype(Ljj_real))0) {
                new_L_diag_recip = (typename CholeskyTraits::RECIP_DIAG_T)
                    ((typename CholeskyTraits::RECIP_DIAG_T)1) /
                    (typename CholeskyTraits::RECIP_DIAG_T)Ljj_real;
            } else {
                new_L_diag_recip = 0;
            }
        }
        // Store diagonal value
        if (LowerTriangularL == true) {
            L[j][j] = new_L;
        } else {
            L[j][j] = hls::x_conj(new_L);
        }

    sum_loop:
        for (int k = 0; k <= j; k++) {
// Define average trip count for reporting, loop reduces in length for every iteration of col_loop
#pragma HLS loop_tripcount max = 1 + RowsColsA / 2
            // Same value used in all calcs
            // o Implement -1* here
            prod_column_top = -hls::x_conj(L_internal[j][k]);

        // NOTE: Using a fixed loop length combined with a "if" to implement reducing loop length
        // o Ensures the inner loop can achieve the maximum II (1)
        // o May introduce a small overhead resolving the "if" statement but HLS struggled to schedule when the variable
        //   loop bound expression was used.
        // o Will report inaccurate trip count as it will reduce by one with the col_loop
        // o Variable loop bound code: row_loop: for(int i = j+1; i < RowsColsA; i++) {
        row_loop:
            for (int i = 0; i < RowsColsA; i++) {
// IMPORTANT: row_loop must not merge with sum_loop as the merged loop becomes variable length and HLS will struggle
// with scheduling
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
#pragma HLS UNROLL FACTOR = CholeskyTraits::UNROLL_FACTOR

                if (i > j) {
                    prod = L_internal[i][k] * prod_column_top;
                    prod_cast_to_sum = prod;

                    if (k == 0) {
                        // Prime first sum
                        if (LowerTriangularL == true) {
                            A_cast_to_sum = A[i][j];
                        } else {
                            // Upper-triangular: prime with A(j,i) directly (no conjugation)
                            A_cast_to_sum = A[j][i];
                        }
                        product_sum = A_cast_to_sum;
                    } else {
                        product_sum = product_sum_array[i];
                    }

                    if (k < j) {
                        // Accumulate row sum of columns
                        product_sum_array[i] = product_sum + prod_cast_to_sum;
                    } else {
                        // Final calculation for off diagonal value
                        prod_cast_to_off_diag = product_sum;
                        // Diagonal is stored in its reciprocal form so only need to multiply the product sum
                        cholesky_prod_sum_mult(prod_cast_to_off_diag, new_L_diag_recip, new_L_off_diag);
                        // Round to target format using method specifed by traits defined types.
                        new_L = new_L_off_diag;
                        // Build sum for use in diagonal calculation (accumulate |L(i,j)|^2)
                        if (k == 0) {
                            square_sum_array[j] = hls::x_conj(new_L) * new_L;
                        } else {
                            square_sum_array[j] += hls::x_conj(new_L) * new_L;
                        }
                        // Store result
                        L_internal[i][j] = new_L;
                        // NOTE: Use the upper/lower triangle zeroing in the subsequent loop so the double memory access
                        // does not
                        // become a bottleneck
                        // o Results in a further increase of DSP resources due to the higher II of this loop.
                        // o Retaining the zeroing operation here can give this a loop a max II of 2 and HLS will
                        // resource share.
                        if (LowerTriangularL == true) {
                            L[i][j] = new_L;                                   // Store in lower triangle
                            if (!CholeskyTraits::ARCH2_ZERO_LOOP) L[j][i] = 0; // Zero upper
                        } else {
                            // Upper-triangular: store U[j,i] directly without conjugation
                            L[j][i] = new_L;                                   // Store in upper triangle
                            if (!CholeskyTraits::ARCH2_ZERO_LOOP) L[i][j] = 0; // Zero lower
                        }
                    }
                }
            }
        }
    }
    // Zero upper/lower triangle
    // o Use separate loop to ensure main calcuation can achieve an II of 1
    // o As noted above this may increase the DSP resources.
    // o Required when unrolling the inner loop due to array dimension access
    if (CholeskyTraits::ARCH2_ZERO_LOOP) {
    zero_rows_loop:
        for (int i = 0; i < RowsColsA - 1; i++) {
        zero_cols_loop:
            for (int j = i + 1; j < RowsColsA; j++) {
// Define average trip count for reporting, loop reduces in length for every iteration of zero_rows_loop
#pragma HLS loop_tripcount max = 1 + RowsColsA / 2
#pragma HLS PIPELINE
                if (LowerTriangularL == true) {
                    L[i][j] = 0; // Zero upper
                } else {
                    L[j][i] = 0; // Zero lower
                }
            }
        }
    }
    return (return_code);
}

// ===================================================================================================================
// choleskyTop: Top level function that selects implementation architecture and internal types based on the
// traits class provided via the CholeskyTraits template parameter.
// o Call this function directly if you wish to override the default architecture choice or internal types
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyTop(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    switch (CholeskyTraits::ARCH) {
        case 0:
            return choleskyBasic<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        case 1:
            return choleskyAlt<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        case 2:
            return choleskyAlt2<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        default:
            return choleskyBasic<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
    }
}

/**
* @brief cholesky
*
* @tparam LowerTriangularL   When false generates the result in the upper triangle
* @tparam RowsColsA          Defines the matrix dimensions
* @tparam InputType          Input data type
* @tparam OutputType         Output data type
* @tparam TRAITS             choleskyTraits class
*
* @param matrixAStrm         Stream of Hermitian/symmetric positive definite input matrix
* @param matrixLStrm         Stream of Lower or upper triangular output matrix
*
* @return                    An integer type. 0=Success. 1=Failure. The function attempted to find the square root of
* a negative number i.e. the input matrix A was not Hermitian/symmetric positive definite.
*/
template <bool LowerTriangularL,
          int RowsColsA,
          class InputType,
          class OutputType,
          typename TRAITS = choleskyTraits<LowerTriangularL, RowsColsA, InputType, OutputType> >
int cholesky(hls::stream<InputType>& matrixAStrm, hls::stream<OutputType>& matrixLStrm) {
    InputType A[RowsColsA][RowsColsA];
    OutputType L[RowsColsA][RowsColsA];

    for (int r = 0; r < RowsColsA; r++) {
#pragma HLS PIPELINE
        for (int c = 0; c < RowsColsA; c++) {
            matrixAStrm.read(A[r][c]);
        }
    }

    int ret = 0;
    ret = choleskyTop<LowerTriangularL, RowsColsA, TRAITS, InputType, OutputType>(A, L);

    for (int r = 0; r < RowsColsA; r++) {
#pragma HLS PIPELINE
        for (int c = 0; c < RowsColsA; c++) {
            matrixLStrm.write(L[r][c]);
        }
    }
    return ret;
}

} // end namespace solver
} // end namespace xf
#endif
