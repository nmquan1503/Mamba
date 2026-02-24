#pragma once

#include <c10/cuda/CUDAException.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <ATen/cuda/Atomic.cuh>

#include "selective_scan.h"
#include "common.h"
#include "reverse_scan.cuh"
#include "kernel_config.cuh"
#include "static_switch.h"


template<
    bool kSeqDivisible_,
    bool kIsConstantB_,
    bool kIsConstantC_,
    int kMaxStateDim_,
    bool kUseDeltaSoftplus_
> struct BackwardKernelTraits {
    using input_t = float;
    using weight_t = float;
    using state_t = float2;

    static constexpr int kNumThreads = kernel_config::num_threads;
    static constexpr int kMinBlocks = kNumThreads < 128 ? 5 : 3;
    static constexpr int kNumElements = kernel_config::num_elements;
    static constexpr int kElementSizeInBytes = sizeof(input_t);
    static constexpr int kVectorSizeInBytes = 16;
    static constexpr int kVectorSize = kVectorSizeInBytes / kElementSizeInBytes;
    static constexpr int kNumVectors = kNumElements / kVectorSize;
    static constexpr bool kSeqDivisible = kSeqDivisible_;
    static constexpr bool kIsConstantB = kIsConstantB_;
    static constexpr bool kIsConstantC = kIsConstantC_;
    static constexpr bool kUseDeltaSoftplus = kUseDeltaSoftplus_;
    static constexpr bool kEnableDirectVectorIO = kSeqDivisible && kNumVectors == 1;
    static constexpr int kMaxStateDim = kMaxStateDim_;

    using vector_t = typename StorageTypeFor<kVectorSizeInBytes>::Type;
   
    using InputScalarBlockLoad = cub::BlockLoad<
        input_t,
        kNumThreads,
        kNumElements,
        cub::BLOCK_LOAD_WARP_TRANSPOSE
    >;

    using InputVectorBlockLoad = cub::BlockLoad<
        vector_t,
        kNumThreads,
        kNumVectors,
        kEnableDirectVectorIO
            ? cub::BLOCK_LOAD_DIRECT
            : cub::BLOCK_LOAD_WARP_TRANSPOSE
    >;

    using WeightScalarBlockLoad = cub::BlockLoad<
        input_t,
        kNumThreads,
        kNumElements,
        cub::BLOCK_LOAD_WARP_TRANSPOSE
    >;

    using WeightVectorBlockLoad = cub::BlockLoad<
        vector_t,
        kNumThreads,
        kNumVectors,
        kEnableDirectVectorIO
            ? cub::BLOCK_LOAD_DIRECT
            : cub::BLOCK_LOAD_WARP_TRANSPOSE
    >;

    using OutputScalarBlockStore = cub::BlockStore<
        input_t,
        kNumThreads,
        kNumElements,
        cub::BLOCK_STORE_WARP_TRANSPOSE
    >;

    using OutputVectorBlockStore = cub::BlockStore<
        vector_t,
        kNumThreads,
        kNumVectors,
        kEnableDirectVectorIO
            ? cub::BLOCK_STORE_DIRECT
            : cub::BLOCK_STORE_WARP_TRANSPOSE
    >;

    using ForwardStateBlockScan = cub::BlockScan<
        state_t,
        kNumThreads,
        cub::BLOCK_SCAN_RAKING
    >;

    using ReverseStateBlockScan = BlockReverseScan<state_t, kNumThreads>;

    using StateBlockReduce = cub::BlockReduce<state_t, kNumThreads>;

    using ScalarBlockReduce = cub::BlockReduce<float, kNumThreads>;

    using WeightBlockExchange = cub::BlockExchange<
        float,
        kNumThreads,
        kNumElements
    >;

    static constexpr int kSMemIOSizeInBytes = max_of({
        sizeof(typename InputScalarBlockLoad::TempStorage),
        sizeof(typename InputVectorBlockLoad::TempStorage),
        std::max(
            sizeof(typename WeightScalarBlockLoad::TempStorage),
            sizeof(typename WeightVectorBlockLoad::TempStorage)
        ) * (2 - int(kIsConstantB) - int(kIsConstantC)),
        sizeof(typename OutputScalarBlockStore::TempStorage),
        sizeof(typename OutputVectorBlockStore::TempStorage)
    });
    static constexpr int kSMemExchangeSizeInBytes = (2 - int(kIsConstantB) - int(kIsConstantC)) * sizeof(typename WeightBlockExchange::TempStorage);
    static constexpr int kSMemReduceSizeInBytes = sizeof(typename StateBlockReduce::TempStorage);
    static constexpr int kSMemSizeInBytes = kSMemIOSizeInBytes 
        + kSMemExchangeSizeInBytes
        + kSMemReduceSizeInBytes
        + sizeof(typename ForwardStateBlockScan::TempStorage)
        + sizeof(typename ReverseStateBlockScan::TempStorage);
};

template<typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads, Traits::kMinBlocks)
void backward_kernel(BackwardParams params) {
    constexpr bool kIsConstantB = Traits::kIsConstantB;
    constexpr bool kIsConstantC = Traits::kIsConstantC;
    constexpr bool kUseDeltaSoftplus = Traits::kUseDeltaSoftplus;
    constexpr int kNumThreads = Traits::kNumThreads;
    constexpr int kNumElements = Traits::kNumElements;
    constexpr int kMaxStateDim = Traits::kMaxStateDim;
    constexpr bool kEnableDirectVectorIO = Traits::kEnableDirectVectorIO;
    using input_t = typename Traits::input_t;
    using weight_t = typename Traits::weight_t;
    using state_t = typename Traits::state_t;

    extern __shared__ char smem_[];

    auto& smem_input_load = reinterpret_cast<
        typename Traits::InputScalarBlockLoad::TempStorage&
    >(smem_);

    auto& smem_weight_load_primary = reinterpret_cast<
        typename Traits::WeightScalarBlockLoad::TempStorage&
    >(smem_);

    auto& smem_weight_load_secondary = *reinterpret_cast<
        typename Traits::WeightScalarBlockLoad::TempStorage*
    >(
        smem_ + sizeof(typename Traits::WeightScalarBlockLoad::TempStorage)
    );

    auto& smem_out_store = reinterpret_cast<
        typename Traits::OutputScalarBlockStore::TempStorage&
    >(smem_);

    auto& smem_weight_exchange_primary = *reinterpret_cast<
        typename Traits::WeightBlockExchange::TempStorage*
    >(
        smem_ + Traits::kSMemIOSizeInBytes
    );

    auto& smem_weight_exchange_secondary = *reinterpret_cast<
        typename Traits::WeightBlockExchange::TempStorage*
    >(
        smem_ + Traits::kSMemIOSizeInBytes + sizeof(typename Traits::WeightBlockExchange::TempStorage)
    );

    auto& smem_state_reduce = *reinterpret_cast<
        typename Traits::StateBlockReduce::TempStorage*
    >(
        reinterpret_cast<char*>(&smem_weight_exchange_primary)
            + Traits::kSMemExchangeSizeInBytes
    );

    auto& smem_scalar_reduce = *reinterpret_cast<
        typename Traits::ScalarBlockReduce::TempStorage*
    >(&smem_state_reduce);

    auto& smem_forward_state_scan = *reinterpret_cast<
        typename Traits::ForwardStateBlockScan::TempStorage*
    >(
        reinterpret_cast<char*>(&smem_state_reduce)
            + Traits::kSMemReduceSizeInBytes
    );

    auto& smem_reverse_state_scan = *reinterpret_cast<
        typename Traits::ReverseStateBlockScan::TempStorage*
    >(
        reinterpret_cast<char*>(&smem_forward_state_scan)
            + sizeof(typename Traits::ForwardStateBlockScan::TempStorage)
    );

    weight_t* smem_delta_a = reinterpret_cast<weight_t*>(smem_ + Traits::kSMemSizeInBytes);
    state_t* smem_running_postfix = reinterpret_cast<state_t*>(smem_delta_a + 2 * kMaxStateDim + kNumThreads);
    weight_t* smem_da = reinterpret_cast<weight_t*>(smem_running_postfix + kMaxStateDim);
    weight_t* smem_dbc = reinterpret_cast<weight_t*>(smem_da + kMaxStateDim);

    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    const int group_id = channel_id / (params.num_channels_per_group);

    input_t* u = reinterpret_cast<input_t*>(params.u_ptr)
        + batch_id * params.u_batch_stride
        + channel_id * params.u_channel_stride;
    
    input_t* delta = reinterpret_cast<input_t*>(params.delta_ptr)
        + batch_id * params.delta_batch_stride
        + channel_id * params.delta_channel_stride;

    input_t* dout = reinterpret_cast<input_t*>(params.dout_ptr)
        + batch_id * params.dout_batch_stride
        + channel_id * params.dout_channel_stride;
    
    weight_t* A = reinterpret_cast<weight_t*>(params.A_ptr)
        + channel_id * params.A_channel_stride;

    weight_t* B_const = reinterpret_cast<weight_t*>(params.B_ptr)
        + channel_id * params.B_channel_stride;

    input_t* B_var = reinterpret_cast<input_t*>(params.B_ptr)
        + batch_id * params.B_batch_stride
        + group_id * params.B_group_stride;

    weight_t* C_const = reinterpret_cast<weight_t*>(params.C_ptr)
        + channel_id * params.C_channel_stride;

    input_t* C_var = reinterpret_cast<input_t*>(params.C_ptr)
        + batch_id * params.C_batch_stride
        + group_id * params.C_group_stride;
    
    weight_t* dA = reinterpret_cast<weight_t*>(params.dA_ptr) 
        + channel_id * params.dA_channel_stride;

    weight_t* dB = reinterpret_cast<weight_t*>(params.dB_ptr)
        + (
            kIsConstantB
                ? channel_id * params.dB_channel_stride
                : batch_id * params.dB_batch_stride + group_id * params.dB_group_stride
        );
    
    weight_t* dC = reinterpret_cast<weight_t*>(params.dC_ptr)
        + (
            kIsConstantC
                ? channel_id * params.dC_channel_stride
                : batch_id * params.dC_batch_stride + group_id * params.dC_group_stride
        );
    
    float* dD = params.dD_ptr == nullptr
        ? nullptr
        : reinterpret_cast<float*>(params.dD_ptr) + channel_id;
    
    float D_val = params.D_ptr == nullptr
        ? 0
        : reinterpret_cast<float*>(params.D_ptr)[channel_id];
    
    float* ddelta_bias = params.ddelta_bias_ptr == nullptr
        ? nullptr
        : reinterpret_cast<float*>(params.ddelta_bias_ptr) + channel_id;
    
    float delta_bias = params.delta_bias_ptr == nullptr
        ? 0
        : reinterpret_cast<float*>(params.delta_bias_ptr)[channel_id];

    state_t* x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<state_t*>(params.x_ptr)
            + (batch_id * params.num_channels + channel_id) * params.num_chunks * params.state_dim;
    
    float dD_val = 0;
    float ddelta_bias_val = 0;

    constexpr int kChunkSize = kernel_config::chunk_size;

    u += (params.num_chunks - 1) * kChunkSize;
    delta += (params.num_chunks - 1) * kChunkSize;
    dout += (params.num_chunks - 1) * kChunkSize;
    B_var += (params.num_chunks - 1) * kChunkSize;
    C_var += (params.num_chunks - 1) * kChunkSize;
    for (int chunk_id = params.num_chunks - 1; chunk_id >= 0; chunk_id--) {
        input_t u_vals[kNumElements];
        input_t delta_vals_load[kNumElements];
        input_t dout_vals_load[kNumElements];
        __syncthreads();
        load_input<Traits>(
            u,
            u_vals,
            smem_input_load,
            params.seq_len - chunk_id * kChunkSize
        );
        u -= kChunkSize;
        __syncthreads();
        load_input<Traits>(
            delta,
            delta_vals_load,
            smem_input_load,
            params.seq_len - chunk_id * kChunkSize
        );
        if constexpr (!kUseDeltaSoftplus) {
            delta -= kChunkSize;
        }
        __syncthreads();
        load_input<Traits>(
            dout, 
            dout_vals_load, 
            smem_input_load, 
            params.seq_len - chunk_id * kChunkSize
        );
        dout -= kChunkSize;

        float dout_vals[kNumElements];
        float delta_vals[kNumElements];

        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            dout_vals[i] = float(dout_vals_load[i]);
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;
            if constexpr (kUseDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f
                    ? log1pf(expf(delta_vals[i]))
                    : delta_vals[i];
            }
        }

        float du_vals[kNumElements];
        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            du_vals[i] = D_val * dout_vals[i];
        }

        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            dD_val += dout_vals[i] * float(u_vals[i]);
        }

        float ddelta_vals[kNumElements] = {0};
        __syncthreads();
        for (int state_id = 0; state_id < params.state_dim; state_id++) {
            const weight_t A_val = A[state_id * params.A_state_stride];
            weight_t A_scaled = A_val * M_LOG2E;
            weight_t B_val, C_val;
            weight_t B_vals[kNumElements], C_vals[kNumElements];
            if constexpr(kIsConstantB) {
                B_val = B_const[state_id * params.B_state_stride];
            }
            else {
                load_weight<Traits>(
                    B_var + state_id * params.B_state_stride,
                    B_vals,
                    smem_weight_load_primary,
                    params.seq_len - chunk_id * kChunkSize
                );
            }
            if constexpr (kIsConstantC) {
                C_val = C_const[state_id * params.C_state_stride];
            }
            else {
                auto& smem_load_weight_C = kIsConstantB
                    ? smem_weight_load_primary
                    : smem_weight_load_secondary;
                load_weight<Traits>(
                    C_var + state_id * params.C_state_stride,
                    C_vals,
                    smem_load_weight_C,
                    params.seq_len - chunk_id * kChunkSize
                );
            }
            state_t states[kNumElements];
            state_t reverse_states[kNumElements];
            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                const float delta_a_exp = exp2f(delta_vals[i] * A_scaled);
                states[i] = make_float2(
                    delta_a_exp,
                    kIsConstantB 
                        ? delta_vals[i] * float(u_vals[i])
                        : delta_vals[i] * float(u_vals[i]) * B_vals[i]
                );

                if (i == 0) {
                    smem_delta_a[
                        threadIdx.x == 0
                            ? state_id + (chunk_id % 2) * kMaxStateDim
                            : threadIdx.x + 2 * kMaxStateDim
                    ] = delta_a_exp;
                }
                else {
                    reverse_states[i - 1].x = delta_a_exp;
                }
                reverse_states[i].y = dout_vals[i] * (
                    kIsConstantC
                        ? (kIsConstantB ? B_val * C_val : C_val)
                        : (kIsConstantB ? B_val * C_vals[i] : C_vals[i])
                );
            }
            __syncthreads();
            reverse_states[kNumElements - 1].x = threadIdx.x == kNumThreads - 1
                ? (chunk_id == params.num_chunks - 1 ? 1.f : smem_delta_a[state_id + ((chunk_id + 1) % 2) * kMaxStateDim])
                : smem_delta_a[threadIdx.x + 1 + 2 * kMaxStateDim];
            
            state_t running_prefix = chunk_id > 0 && threadIdx.x % 32 == 0
                ? x[(chunk_id - 1) * params.state_dim + state_id]
                : make_float2(1.f, 0.f);
            
            ScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
            typename Traits::ForwardStateBlockScan(smem_forward_state_scan).InclusiveScan(
                states,
                states,
                ScanOp<weight_t>(),
                prefix_op
            );
            state_t running_postfix = chunk_id < params.num_chunks - 1 && threadIdx.x % 32 == 0
                ? smem_running_postfix[state_id]
                : make_float2(1.f, 0.f);
            
            ScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
            typename Traits::ReverseStateBlockScan(smem_reverse_state_scan).InclusiveReverseScan(
                reverse_states,
                reverse_states,
                ScanOp<weight_t>(),
                postfix_op
            );
            if (threadIdx.x == 0) {
                smem_running_postfix[state_id] = postfix_op.running_prefix;
            }
            weight_t dA_val = 0, dBC_val = 0;
            weight_t dB_vals[kNumElements], dC_vals[kNumElements];
            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                const float dx = reverse_states[i].y;
                const float ddelta_u = kIsConstantB ? dx : dx * B_vals[i];
                du_vals[i] += ddelta_u * delta_vals[i];
                const float a = states[i].y - (kIsConstantB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]);
                ddelta_vals[i] += ddelta_u * float(u_vals[i]) + dx * A_val * a;
                dA_val += dx * delta_vals[i] * a;
                if constexpr (kIsConstantB || kIsConstantC) {
                    if constexpr (kIsConstantB) {
                        dBC_val += dout_vals[i] * (kIsConstantC ? states[i].y : states[i].y * C_vals[i]);
                    }
                    else {
                        dBC_val += dout_vals[i] * states[i].y;
                    }
                }
                if constexpr (!kIsConstantB) {
                    dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]);
                }
                if constexpr (!kIsConstantC) {
                    dC_vals[i] = dout_vals[i] * (kIsConstantB ? states[i].y * B_val : states[i].y);
                }
            }
            if constexpr ((!kIsConstantB) || (!kIsConstantC)) {
                if constexpr (!kIsConstantB) {
                    typename Traits::WeightBlockExchange(smem_weight_exchange_primary).BlockedToStriped(dB_vals, dB_vals);
                }
                if constexpr (!kIsConstantC) {
                    auto& smem_exchange_C = kIsConstantB ? smem_weight_exchange_primary : smem_weight_exchange_secondary;
                    typename Traits::WeightBlockExchange(smem_exchange_C).BlockedToStriped(dC_vals, dC_vals);
                }
                const int seqlen_remaining = params.seq_len - chunk_id * kChunkSize - threadIdx.x;
                weight_t* dB_cur = dB + state_id * params.dB_state_stride + chunk_id * kChunkSize + threadIdx.x;
                weight_t* dC_cur = dC + state_id * params.dC_state_stride + chunk_id * kChunkSize + threadIdx.x;
                #pragma unroll
                for (int i = 0; i < kNumElements; i++) {
                    if (i * kNumThreads < seqlen_remaining) {
                        if constexpr (!kIsConstantB) {
                            gpuAtomicAdd(dB_cur + i * kNumThreads, dB_vals[i]);
                        }
                        if constexpr (!kIsConstantC) {
                            gpuAtomicAdd(dC_cur + i * kNumThreads, dC_vals[i]);
                        }
                    }
                }
            }
            if constexpr (kIsConstantB || kIsConstantC) {
                float2 dA_dBC_val = make_float2(dA_val, dBC_val);
                dA_dBC_val = typename Traits::StateBlockReduce(smem_state_reduce).Sum(dA_dBC_val);
                dA_val = dA_dBC_val.x;
                if (threadIdx.x == 0) {
                    smem_dbc[state_id] = chunk_id == params.num_chunks - 1
                        ? dA_dBC_val.y
                        : dA_dBC_val.y + smem_dbc[state_id];
                }
            }
            else {
                dA_val = typename Traits::ScalarBlockReduce(smem_scalar_reduce).Sum(dA_val);
            }
            if (threadIdx.x == 0) {
                smem_da[state_id] = chunk_id == params.num_chunks - 1
                    ? dA_val
                    : dA_val + smem_da[state_id];
            }
        }
        if (kUseDeltaSoftplus) {
            __syncthreads();
            input_t delta_vals_load[kNumElements];
            load_input<Traits>(
                delta,
                delta_vals_load,
                smem_input_load,
                params.seq_len - chunk_id * kChunkSize
            );
            delta -= kChunkSize;
            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                float delta_val = float(delta_vals_load[i]) + delta_bias;
                float delta_val_neg_exp = expf(-delta_val);
                ddelta_vals[i] = delta_val <= 20.f
                    ? ddelta_vals[i] / (1.f + delta_val_neg_exp)
                    : ddelta_vals[i];
            }
        }

        for (int i = 0; i < kNumElements; i++) {
            ddelta_bias_val += ddelta_vals[i];
        }

        input_t* du = reinterpret_cast<input_t*>(params.du_ptr)
            + batch_id * params.du_batch_stride
            + channel_id * params.du_channel_stride
            + chunk_id * kChunkSize;
        
        input_t* ddelta = reinterpret_cast<input_t*>(params.ddelta_ptr)
            + batch_id * params.ddelta_batch_stride
            + channel_id * params.ddelta_channel_stride
            + chunk_id * kChunkSize;
        __syncthreads();
        store_output<Traits>(
            du,
            du_vals,
            smem_out_store,
            params.seq_len - chunk_id * kChunkSize
        );

        __syncthreads();
        store_output<Traits>(
            ddelta,
            ddelta_vals,
            smem_out_store,
            params.seq_len - chunk_id * kChunkSize
        );
        B_var -= kChunkSize;
        C_var -= kChunkSize;
    }
    if (params.dD_ptr != nullptr) {
        dD_val = typename Traits::ScalarBlockReduce(smem_scalar_reduce).Sum(dD_val);
        if (threadIdx.x == 0) {
            gpuAtomicAdd(dD, dD_val);
        }
    }
    if (params.ddelta_bias_ptr != nullptr) {
        __syncthreads();
        ddelta_bias_val = typename Traits::ScalarBlockReduce(smem_scalar_reduce).Sum(ddelta_bias_val);
        if (threadIdx.x == 0) {
            gpuAtomicAdd(ddelta_bias, ddelta_bias_val);
        } 
    }
    for (int state_id = threadIdx.x; state_id < params.state_dim; state_id += blockDim.x) {
        gpuAtomicAdd(&(dA[state_id * params.dA_state_stride]), smem_da[state_id]);
        weight_t dBC_val;
        if (kIsConstantB || kIsConstantC) {
            dBC_val = smem_dbc[state_id];
        }
        if constexpr (kIsConstantB) {
            gpuAtomicAdd(
                &(dB[state_id * params.dB_state_stride]), 
                kIsConstantC 
                    ? dBC_val * C_const[state_id * params.C_state_stride]
                    : dBC_val
            );
        }
        if constexpr (kIsConstantC) {
            gpuAtomicAdd(
                &(dC[state_id * params.dC_state_stride]),
                kIsConstantB
                    ? dBC_val * B_const[state_id * params.B_state_stride]
                    : dBC_val
            );
        }
    }
}

void backward_kernel_launch(BackwardParams& params, cudaStream_t stream) {
    BOOL_SWITCH(params.seq_len % kernel_config::chunk_size == 0, kSeqDivisible, [&]{
        BOOL_SWITCH(params.is_constant_B, kIsConstantB, [&]{
            BOOL_SWITCH(params.is_constant_C, kIsConstantC, [&]{
                BOOL_SWITCH(params.use_delta_softplus, kUseDeltaSoftplus, [&]{
                    DISPATCH_SWITCH(params.state_dim, MAX_STATE_DIM, [&]{
                        using Traits = BackwardKernelTraits<
                            kSeqDivisible, 
                            kIsConstantB,
                            kIsConstantC, 
                            MAX_STATE_DIM,
                            kUseDeltaSoftplus
                        >;
                        constexpr int kSMemSizeInBytes = Traits::kSMemSizeInBytes 
                            + MAX_STATE_DIM * sizeof(typename Traits::state_t)
                            + (kernel_config::num_threads + 4 * MAX_STATE_DIM) * sizeof(Traits::weight_t);
                        dim3 grid(params.batch_size, params.num_channels);
                        auto kernel = &backward_kernel<Traits>;
                        if (kSMemSizeInBytes >= 48 * 1024) {
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMemSizeInBytes
                            ));
                        }
                        kernel<<<grid, Traits::kNumThreads, kSMemSizeInBytes, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });        
        });
    });
}