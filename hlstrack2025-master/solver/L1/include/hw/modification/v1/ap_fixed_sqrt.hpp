#ifndef AP_FIXED_SQRT_FAST_HPP
#define AP_FIXED_SQRT_FAST_HPP

#include "ap_fixed.h"
#include "hls_math.h"
#include "hls_stream.h"

template<int LUT_BITS = 6>
struct SqrtParams {
    enum { LUT_SIZE = 1 << LUT_BITS };
    static const int LUT = LUT_BITS;
};
static const ap_ufixed<24,1> sqrt_lut_64[64] = {
    (ap_ufixed<24,1>)1.397926490657660725e+00,
    (ap_ufixed<24,1>)1.366972252346817518e+00,
    (ap_ufixed<24,1>)1.337987216011345293e+00,
    (ap_ufixed<24,1>)1.310771072830464901e+00,
    (ap_ufixed<24,1>)1.285150926243998182e+00,
    (ap_ufixed<24,1>)1.260976649982560982e+00,
    (ap_ufixed<24,1>)1.238117172054618909e+00,
    (ap_ufixed<24,1>)1.216457474031528818e+00,
    (ap_ufixed<24,1>)1.195896148403097436e+00,
    (ap_ufixed<24,1>)1.176343395350092358e+00,
    (ap_ufixed<24,1>)1.157719368467874599e+00,
    (ap_ufixed<24,1>)1.139952799806554395e+00,
    (ap_ufixed<24,1>)1.122979850149754322e+00,
    (ap_ufixed<24,1>)1.106743142185719453e+00,
    (ap_ufixed<24,1>)1.091190943152757553e+00,
    (ap_ufixed<24,1>)1.076276470394099904e+00,
    (ap_ufixed<24,1>)1.061957298559955243e+00,
    (ap_ufixed<24,1>)1.048194851328669408e+00,
    (ap_ufixed<24,1>)1.034953963765041340e+00,
    (ap_ufixed<24,1>)1.022202503999903866e+00,
    (ap_ufixed<24,1>)1.009911044956484982e+00,
    (ap_ufixed<24,1>)9.980525784828885305e-01,
    (ap_ufixed<24,1>)9.866022655651317530e-01,
    (ap_ufixed<24,1>)9.755372173595077134e-01,
    (ap_ufixed<24,1>)9.648363026488434580e-01,
    (ap_ufixed<24,1>)9.544799780350297080e-01,
    (ap_ufixed<24,1>)9.444501377615284188e-01,
    (ap_ufixed<24,1>)9.347299805391775518e-01,
    (ap_ufixed<24,1>)9.253038911459843252e-01,
    (ap_ufixed<24,1>)9.161573349021892021e-01,
    (ap_ufixed<24,1>)9.072767633979883506e-01,
    (ap_ufixed<24,1>)8.986495300827685995e-01,
    (ap_ufixed<24,1>)8.902638145194515795e-01,
    (ap_ufixed<24,1>)8.821085542719540040e-01,
    (ap_ufixed<24,1>)8.741733835330449676e-01,
    (ap_ufixed<24,1>)8.664485777182117099e-01,
    (ap_ufixed<24,1>)8.589250033520303695e-01,
    (ap_ufixed<24,1>)8.515940726597591715e-01,
    (ap_ufixed<24,1>)8.444477023508152325e-01,
    (ap_ufixed<24,1>)8.374782761443420043e-01,
    (ap_ufixed<24,1>)8.306786106418639903e-01,
    (ap_ufixed<24,1>)8.240419241993676147e-01,
    (ap_ufixed<24,1>)8.175618084921536521e-01,
    (ap_ufixed<24,1>)8.112322025014301330e-01,
    (ap_ufixed<24,1>)8.050473686826041808e-01,
    (ap_ufixed<24,1>)7.990018711022757181e-01,
    (ap_ufixed<24,1>)7.930905553545753994e-01,
    (ap_ufixed<24,1>)7.873085300881966786e-01,
    (ap_ufixed<24,1>)7.816511499936503737e-01,
    (ap_ufixed<24,1>)7.761140001162655233e-01,
    (ap_ufixed<24,1>)7.706928813745408391e-01,
    (ap_ufixed<24,1>)7.653837971759037684e-01,
    (ap_ufixed<24,1>)7.601829410329278280e-01,
    (ap_ufixed<24,1>)7.550866850928139584e-01,
    (ap_ufixed<24,1>)7.500915695015926143e-01,
    (ap_ufixed<24,1>)7.451942925321957123e-01,
    (ap_ufixed<24,1>)7.403917014123960749e-01,
    (ap_ufixed<24,1>)7.356807837947245687e-01,
    (ap_ufixed<24,1>)7.310586598159261040e-01,
    (ap_ufixed<24,1>)7.265225746983997590e-01,
    (ap_ufixed<24,1>)7.220698918504386832e-01,
    (ap_ufixed<24,1>)7.176980864260105175e-01,
    (ap_ufixed<24,1>)7.134047393083357003e-01,
    (ap_ufixed<24,1>)7.091875314846980416e-01
};

template<int MID_W, int MID_I, int LUT_BITS = 6>
inline ap_fixed<MID_W, MID_I> estimate_fast(ap_fixed<MID_W, MID_I> x) {
    #pragma HLS RESOURCE variable=sqrt_lut_64 core=ROM_1P
    #pragma HLS ARRAY_PARTITION variable=sqrt_lut_64 complete
    typedef ap_fixed<MID_W, MID_I> mid_type;
    static_assert(LUT_BITS == 6, "ap_fixed_sqrt: Only LUT_BITS=6 is supported with the built-in 64-entry LUT");
    const int LUT_SIZE = SqrtParams<LUT_BITS>::LUT_SIZE;
    const mid_type RECIP = (mid_type)0.6666666666666666;
    const mid_type SUB   = (mid_type)0.3333333333333333;
    mid_type idx = x * RECIP - SUB;
    if (idx < (mid_type)0) idx = (mid_type)0;
    if (idx > (mid_type)0.9999999) idx = (mid_type)0.9999999;
    mid_type t = idx * (mid_type)(LUT_SIZE - 1);
    int index = (int)t;
    if (index < 0) index = 0;
    if (index >= LUT_SIZE) index = LUT_SIZE - 1;
    mid_type y = (mid_type)sqrt_lut_64[index];
    return y;
}
template<int W, int I, int LUT_BITS = 6, int ITER = 2>
ap_fixed<W, I> fixed_sqrt_fast(ap_fixed<W, I> x_in) {
    typedef ap_fixed<W, I> in_type;
    const int MID_W = (W*2 > 32) ? 32 : W*2;
    const int MID_I = (I*2 > 12) ? 12 : I*2;
    typedef ap_fixed<MID_W, MID_I> mid_type;
    #pragma HLS INLINE
    if (x_in <= (in_type)0) return (in_type)0;
    mid_type x = (mid_type)x_in;
    int exp = 0;
    const mid_type HALF = (mid_type)0.5;
    const mid_type TWO  = (mid_type)2.0;
    while (x < HALF) { x *= (mid_type)2.0; exp--; }
    while (x >= TWO) { x *= (mid_type)0.5; exp++; }
    mid_type y = estimate_fast<MID_W, MID_I, LUT_BITS>(x);
    for (int it = 0; it < ITER; ++it) {
        #pragma HLS PIPELINE II=1
        mid_type div = x / y;
        y = ( (mid_type)0.5 ) * ( y + div );
    }
    if (exp != 0) {
        int e = exp;
        if (e & 1) {
            y *= (mid_type)1.4142135623730951;
            e--;
        }
        int shift = e >> 1;
        while (shift > 0) { y *= (mid_type)2.0; --shift; }
        while (shift < 0) { y *= (mid_type)0.5; ++shift; }
    }
    in_type out = (in_type)y;
    return out;
}
template<int W, int I, int LUT_BITS = 6, int ITER = 2>
ap_fixed<W, I> fixed_rsqrt_fast(ap_fixed<W, I> x_in) {
    typedef ap_fixed<W, I> in_type;
    const int MID_W = (W*2 > 32) ? 32 : W*2;
    const int MID_I = (I*2 > 12) ? 12 : I*2;
    typedef ap_fixed<MID_W, MID_I> mid_type;

    #pragma HLS INLINE
    if (x_in <= (in_type)0) return (in_type)0;
    mid_type x = (mid_type)x_in;

    int exp = 0;
    const mid_type HALF = (mid_type)0.5;
    const mid_type TWO  = (mid_type)2.0;
    while (x < HALF) { x *= (mid_type)2.0; exp--; }
    while (x >= TWO) { x *= (mid_type)0.5; exp++; }
    mid_type y = estimate_fast<MID_W, MID_I, LUT_BITS>(x);
    y = (mid_type)1.0 / y;
    const mid_type C1 = (mid_type)1.5;
    const mid_type C2 = (mid_type)0.5;
    for (int it = 0; it < ITER; ++it) {
        #pragma HLS PIPELINE II=1
        mid_type y2 = y * y;
        mid_type t = C1 - (C2 * x * y2);
        y = y * t;
    }
    if (exp != 0) {
        int e = exp;
        if (e & 1) {
            y *= (mid_type)0.7071067811865476;
            e--;
        }
        int shift = e >> 1;
        while (shift > 0) { y /= (mid_type)2.0; --shift; }
        while (shift < 0) { y *= (mid_type)2.0; ++shift; }
    }
    in_type out = (in_type)y;
    return out;
}
#endif 
