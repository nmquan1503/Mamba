#include <c10/cuda/CUDAException.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "selective_scan.h"
#include "common.h"
#include "kernel_config.cuh"
#include "static_switch.h"

template<
    bool kSeqDivisible_,
    bool kIsConstantB_,
    bool kIsConstantC_
>
struct KernelTraits {
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
    static constexpr bool kEnableDirectVectorIO = kSeqDivisible && kNumVectors == 1;

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

    using StateBlockScan = cub::BlockScan<
        state_t,
        kNumThreads,
        cub::BLOCK_SCAN_WARP_SCANS
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

    static constexpr int kSMemSizeInBytes = kSMemIOSizeInBytes + sizeof(typename StateBlockScan::TempStorage);
};


template<typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads, Traits::kMinBlocks)
void forward_kernel(ForwardParams params) {
    constexpr bool kIsConstantB = Traits::kIsConstantB;
    constexpr bool kIsConstantC = Traits::kIsConstantC;
    constexpr int kNumThreads = Traits::kNumThreads;
    constexpr int kNumElements = Traits::kNumElements;
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

    auto& smem_scan = *reinterpret_cast<
        typename Traits::StateBlockScan::TempStorage*
    >(smem_ + Traits::kSMemIOSizeInBytes);

    state_t *smem_running_prefix = reinterpret_cast<state_t*>(smem_ + Traits::kSMemSizeInBytes);

    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    const int group_id = channel_id / (params.num_channels_per_group);
 
    input_t* u = reinterpret_cast<input_t*>(params.u_ptr) 
        + batch_id * params.u_batch_stride
        + channel_id * params.u_channel_stride;
    
    input_t* delta = reinterpret_cast<input_t*>(params.delta_ptr)
        + batch_id * params.delta_batch_stride
        + channel_id * params.delta_channel_stride;
    
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

    state_t* x = reinterpret_cast<state_t*>(params.x_ptr)
        + (batch_id * params.num_channels + channel_id) * params.num_chunks * params.state_dim;

    float D_val = 0.f;
    if (params.D_ptr != nullptr) {
        D_val = reinterpret_cast<float*>(params.D_ptr)[channel_id];
    }

    float delta_bias = 0.f;
    if (params.delta_bias_ptr != nullptr) {
        delta_bias = reinterpret_cast<float*>(params.delta_bias_ptr)[channel_id];
    }

    constexpr int kChunkSize = kernel_config::chunk_size;

    for (int chunk_id = 0; chunk_id < params.num_chunks; chunk_id++) {
        input_t u_vals[kNumElements];
        input_t delta_vals_load[kNumElements];
        __syncthreads();

        load_input<Traits>(
            u,
            u_vals,
            smem_input_load,
            params.seq_len - chunk_id * kChunkSize
        );
        if constexpr (!kEnableDirectVectorIO) {
            __syncthreads();
        }
        load_input<Traits>(
            delta,
            delta_vals_load,
            smem_input_load,
            params.seq_len - chunk_id * kChunkSize
        );

        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNumElements];
        float delta_u_vals[kNumElements];
        float out_vals[kNumElements];
        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            float u_val = float(u_vals[i]);
            delta_vals[i] = float(delta_vals_load[i] + delta_bias);
            if (params.use_delta_softplus) {
                delta_vals[i] = delta_vals[i] <= 20.f 
                    ? log1pf(expf(delta_vals[i])) 
                    : delta_vals[i];
            }
            delta_u_vals[i] = delta_vals[i] * u_val;
            out_vals[i] = D_val * u_val;
        }

        __syncthreads();
        for (int state_id = 0; state_id < params.state_dim; state_id++) {
            weight_t A_val = A[state_id * params.A_state_stride] * M_LOG2E;

            weight_t BC_val;
            weight_t B_vals[kNumElements];
            weight_t C_vals[kNumElements];
            if constexpr (!kIsConstantB) {
                load_weight<Traits>(
                    B_var + state_id * params.B_state_stride,
                    B_vals,
                    smem_weight_load_primary,
                    params.seq_len - chunk_id * kChunkSize
                );
                if constexpr (kIsConstantC) {
                    BC_val = C_const[state_id * params.C_state_stride];
                }
            }
            if constexpr (!kIsConstantC) {
                auto &smem_load_weight_C = kIsConstantB ? smem_weight_load_primary : smem_weight_load_secondary;
                load_weight<Traits>(
                    C_var + state_id * params.C_state_stride,
                    C_vals,
                    smem_load_weight_C,
                    params.seq_len - chunk_id * kChunkSize
                );
                if constexpr (kIsConstantB) {
                    BC_val = B_const[state_id * params.B_state_stride];
                }
            }
            if constexpr (kIsConstantB && kIsConstantC) {
                BC_val = B_const[state_id * params.B_state_stride] * C_const[state_id * params.C_state_stride];
            }

            state_t states[kNumElements];
            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                states[i] = make_float2(
                    exp2f(delta_vals[i] * A_val),
                    kIsConstantB ? delta_u_vals[i] : B_vals[i] * delta_u_vals[i]
                );
                if constexpr (!Traits::kSeqDivisible) {
                    if (threadIdx.x * kNumElements + i >= params.seq_len - chunk_id * kChunkSize) {
                        states[i] = make_float2(1.f, 0.f);
                    }
                }
            }

            state_t running_prefix;
            running_prefix = chunk_id > 0 && threadIdx.x % 32 == 0
                ? smem_running_prefix[state_id]
                : make_float2(1.f, 0.f);
            
            ScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
            typename Traits::StateBlockScan(smem_scan).InclusiveScan(
                states, states, ScanOp<weight_t>(), prefix_op
            );
            if (threadIdx.x == 0) {
                smem_running_prefix[state_id] = prefix_op.running_prefix;
                x[chunk_id * params.state_dim + state_id] = prefix_op.running_prefix;
            }
            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                const weight_t C_val = kIsConstantC
                    ? BC_val
                    : (kIsConstantB ? BC_val *  C_vals[i] : C_vals[i]);
                out_vals[i] += states[i].y * C_val;
            }
        }
        input_t* out = reinterpret_cast<input_t*>(params.out_ptr)
            + batch_id * params.out_batch_stride
            + channel_id * params.out_channel_stride
            + chunk_id * kChunkSize;
        
        __syncthreads();
        store_output<Traits>(
            out,
            out_vals,
            smem_out_store,
            params.seq_len - chunk_id * kChunkSize
        );

        B_var += kChunkSize;
        C_var += kChunkSize;
    }   
}

void forward_kernel_launch(ForwardParams& params, cudaStream_t stream) {
    BOOL_SWITCH(params.seq_len % kernel_config::chunk_size == 0, kSeqDivisible, [&]{
        BOOL_SWITCH(params.is_constant_B, kIsConstantB, [&]{
            BOOL_SWITCH(params.is_constant_C, kIsConstantC, [&]{
                using Traits = KernelTraits<kSeqDivisible, kIsConstantB, kIsConstantC>;
                DISPATCH_SWITCH(params.state_dim, MAX_STATE_DIM, [&]{
                    constexpr int kSMemSizeInBytes = Traits::kSMemSizeInBytes + MAX_STATE_DIM * sizeof(typename Traits::state_t);
                    dim3 grid(params.batch_size, params.num_channels);
                    auto kernel = &forward_kernel<Traits>;
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
}