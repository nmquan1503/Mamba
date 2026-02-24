#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include <initializer_list>

inline __device__ 
float2 make_ab(float a, float b) {
    return make_float2(a, b);
}

inline __device__ 
float2 operator+(const float2& x, const float2& y) {
    return make_float2(x.x + y.x, x.y + y.y);
}

/////

constexpr size_t max_of(std::initializer_list<size_t> list) {
    auto it = list.begin();
    size_t max_val = *it++;
    for (; it != list.end(); it++) {
        if (*it > max_val) {
            max_val = *it;
        }
    }
    return max_val;
}

/////

template<int BYTES> 
struct StorageTypeFor;

template<> 
struct StorageTypeFor<16> {
    using Type = uint4;
};

template<> 
struct StorageTypeFor<8> {
    using Type = uint64_t;
};

template<> 
struct StorageTypeFor<4> {
    using Type = uint32_t;
};

template<>
struct StorageTypeFor<2> {
    using Type = uint16_t;
};

/////

template<typename scalar_t, int N>
struct Converter {
    static inline __device__ void to_float(const scalar_t (&src)[N], float (&dst)[N]) {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            dst[i] = src[i];
        }
    }
};

/////

template<typename scalar_t> struct ScanOp;

template<>
struct ScanOp<float> {
    __device__ __forceinline__
    float2 operator()(const float2& ab0, const float2& ab1) const {
        float a0 = ab0.x;
        float b0 = ab0.y;
        float a1 = ab1.x;
        float b1 = ab1.y;
        return make_float2(a1 * a0, a1 * b0 + b1);
    }
};

template<typename scalar_t>
struct ScanPrefixCallbackOp {
    using scan_t = std::conditional_t<std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;

    __device__ 
    ScanPrefixCallbackOp(scan_t init) : running_prefix(init) { }

    __device__
    scan_t operator()(scan_t block_aggregate) {
        scan_t old = running_prefix;
        running_prefix = ScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old;
    }
};

/////

template<typename Traits>
inline __device__
void load_input(
    typename Traits::input_t* src,
    typename Traits::input_t (&dst)[Traits::kNumElements],
    typename Traits::InputScalarBlockLoad::TempStorage& smem,
    int seq_len
) {
    if constexpr (Traits::kSeqDivisible) {
        auto& smem_vec = reinterpret_cast<typename Traits::InputVectorBlockLoad::TempStorage&>(smem);
        using vector_t = typename Traits::vector_t;
        typename Traits::InputVectorBlockLoad(smem_vec).Load(
            reinterpret_cast<vector_t*>(src),
            reinterpret_cast<vector_t(&)[Traits::kNumVectors]>(dst)
        );
    }
    else {
        typename Traits::InputScalarBlockLoad(smem).Load(src, dst, seq_len, 0.f);
    }
}

template<typename Traits>
inline __device__
void load_weight(
    typename Traits::input_t* src,
    typename Traits::weight_t (&dst)[Traits::kNumElements],
    typename Traits::WeightScalarBlockLoad::TempStorage &smem,
    int seq_len
) {
    constexpr int kNumElements = Traits::kNumElements;
    typename Traits::input_t vals_load[kNumElements];
    if constexpr (Traits::kSeqDivisible) {
        auto& smem_vec = reinterpret_cast<typename Traits::WeightVectorBlockLoad::TempStorage&>(smem);
        using vector_t = typename Traits::vector_t;
        typename Traits::WeightVectorBlockLoad(smem_vec).Load(
            reinterpret_cast<vector_t*>(src),
            reinterpret_cast<vector_t(&)[Traits::kNumVectors]>(vals_load)
        );
    }
    else {
        typename Traits::WeightScalarBlockLoad(smem).Load(src, vals_load, seq_len, 0.f);
    }
    Converter<typename Traits::input_t, kNumElements>::to_float(vals_load, dst);
}

template<typename Traits>
inline __device__
void store_output(
    typename Traits::input_t* out,
    const float (&out_vals)[Traits::kNumElements],
    typename Traits::OutputScalarBlockStore::TempStorage& smem,
    int seq_len
) {
    typename Traits::input_t vals[Traits::kNumElements];
    #pragma unroll
    for (int i = 0; i < Traits::kNumElements; i++) {
        vals[i] = out_vals[i];
    }
    if constexpr (Traits::kSeqDivisible) {
        auto& smem_vec = reinterpret_cast<typename Traits::OutputVectorBlockStore::TempStorage&>(smem);
        using vector_t = typename Traits::vector_t;
        typename Traits::OutputVectorBlockStore(smem_vec).Store(
            reinterpret_cast<vector_t*>(out),
            reinterpret_cast<vector_t(&)[Traits::kNumVectors]>(vals)
        );
    }
    else {
        typename Traits::OutputScalarBlockStore(smem).Store(out, vals, seq_len);
    }
}