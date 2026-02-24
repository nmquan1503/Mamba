#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>
#include <vector>

#include "selective_scan.h"
#include "kernel_config.cuh"

void forward_kernel_launch(ForwardParams &params, cudaStream_t stream);
void backward_kernel_launch(BackwardParams &params, cudaStream_t stream);

void set_forward_params(
    ForwardParams &params, 
    const size_t batch_size,
    const size_t seq_len,
    const size_t state_dim,
    const size_t num_channels,
    const size_t num_groups,
    const size_t num_chunks,
    const bool is_constant_B,
    const bool is_constant_C,
    const at::Tensor u,
    const at::Tensor delta,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor C,
    const at::Tensor out,
    void* D_ptr,
    void* delta_bias_ptr,
    void* x_ptr,
    bool use_delta_softplus
) {
    memset(&params, 0, sizeof(params));

    params.batch_size = batch_size;
    params.seq_len = seq_len;
    params.state_dim = state_dim;
    params.num_channels = num_channels;
    params.num_groups = num_groups;
    params.num_chunks = num_chunks;
    params.num_channels_per_group = num_channels / num_groups;

    params.use_delta_softplus = use_delta_softplus;
    params.is_constant_B = is_constant_B;
    params.is_constant_C = is_constant_C;

    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.x_ptr = x_ptr;

    params.A_channel_stride = A.stride(0);
    params.A_state_stride = A.stride(1);

    if (is_constant_B) {
        params.B_channel_stride = B.stride(0);
    }
    else {
        params.B_batch_stride = B.stride(0);
        params.B_group_stride = B.stride(1);
    }
    params.B_state_stride = is_constant_B ? B.stride(1) : B.stride(2);
    
    if (is_constant_C) {
        params.C_channel_stride = C.stride(0);
    }
    else {
        params.C_batch_stride = C.stride(0);
        params.C_group_stride = C.stride(1);
    }
    params.C_state_stride = is_constant_C ? C.stride(1) : C.stride(2);

    params.u_batch_stride = u.stride(0);
    params.u_channel_stride = u.stride(1);

    params.delta_batch_stride = delta.stride(0);
    params.delta_channel_stride = delta.stride(1);

    params.out_batch_stride = out.stride(0);
    params.out_channel_stride = out.stride(1);
}

void set_backward_params(
    BackwardParams &params,
    const size_t batch_size,
    const size_t num_channels,
    const size_t seq_len,
    const size_t state_dim,
    const size_t num_groups,
    const size_t num_chunks,
    const size_t is_constant_B,
    const size_t is_constant_C,
    const at::Tensor u,
    const at::Tensor delta,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor C,
    const at::Tensor out,
    void* D_ptr,
    void* delta_bias_ptr,
    void* x_ptr,
    const at::Tensor dout,
    const at::Tensor du,
    const at::Tensor ddelta,
    const at::Tensor dA,
    const at::Tensor dB,
    const at::Tensor dC,
    void* dD_ptr,
    void* ddelta_bias_ptr,
    bool use_delta_softplus
) {
    set_forward_params(
        params, 
        batch_size, 
        seq_len, 
        state_dim, 
        num_channels,
        num_groups, 
        num_chunks,
        is_constant_B, 
        is_constant_C,
        u, 
        delta, 
        A, 
        B, 
        C,
        out,
        D_ptr, 
        delta_bias_ptr, 
        x_ptr, 
        use_delta_softplus
    );

    params.dout_ptr = dout.data_ptr();
    params.du_ptr = du.data_ptr();
    params.dA_ptr = dA.data_ptr();
    params.dB_ptr = dB.data_ptr();
    params.dC_ptr = dC.data_ptr();
    params.dD_ptr = dD_ptr;
    params.ddelta_ptr = ddelta.data_ptr();
    params.ddelta_bias_ptr = ddelta_bias_ptr;

    params.dout_batch_stride = dout.stride(0);
    params.dout_channel_stride = dout.stride(1);

    params.dA_channel_stride = dA.stride(0);
    params.dA_state_stride = dA.stride(1);

    if (is_constant_B) {
        params.dB_channel_stride = dB.stride(0);
    }
    else {
        params.dB_batch_stride = dB.stride(0);
        params.dB_group_stride = dB.stride(1);
    }
    params.dB_state_stride = is_constant_B ? dB.stride(1) : dB.stride(2);
    
    if (is_constant_C) {
        params.dC_channel_stride = dC.stride(0);
    }
    else {
        params.dC_batch_stride = dC.stride(0);
        params.dC_group_stride = dC.stride(1);
    }
    params.dC_state_stride = is_constant_C ? dC.stride(1) : dB.stride(2);

    params.du_batch_stride = du.stride(0);
    params.du_channel_stride = du.stride(1);
    
    params.ddelta_batch_stride = ddelta.stride(0);
    params.ddelta_channel_stride = ddelta.stride(1);
}

std::vector<at::Tensor> selective_scan_forward(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D_,
    const c10::optional<at::Tensor>& delta_bias_,
    bool use_delta_softplus
) {
    const bool is_constant_B = B.dim() < 3;
    const bool is_constant_C = C.dim() < 3;

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int num_channels = sizes[1];
    const int seq_len = sizes[2];
    const int state_dim = A.size(1);
    const int num_groups = is_constant_B ? 1 : B.size(1);
    const int chunk_size = kernel_config::chunk_size;
    const int num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    at::Tensor out = torch::empty_like(delta);
    at::Tensor x = torch::empty({
        batch_size, 
        num_channels, 
        num_chunks, 
        state_dim * 2
    }, u.options());

    ForwardParams params;
    set_forward_params(
        params,
        batch_size, 
        seq_len, 
        state_dim, 
        num_channels, 
        num_groups, 
        num_chunks,
        is_constant_B, 
        is_constant_C,
        u, 
        delta, 
        A, 
        B, 
        C, 
        out,
        D_.has_value() ? D_.value().data_ptr() : nullptr,
        delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
        x.data_ptr(), 
        use_delta_softplus
    );

    at::cuda::CUDAGuard device_goard(u.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    forward_kernel_launch(params, stream);

    std::vector<at::Tensor> result = {out, x};
    return result;
}

std::vector<at::Tensor> selective_scan_backward(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D_,
    const c10::optional<at::Tensor>& delta_bias_,
    const at::Tensor& dout,
    const c10::optional<at::Tensor>& x_,
    // const c10::optional<at::Tensor>& out_,
    bool use_delta_softplus
) {
    const bool is_constant_B = B.dim() < 3;
    const bool is_constant_C = C.dim() < 3;

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int num_channels = sizes[1];
    const int seq_len = sizes[2];
    const int state_dim = A.size(1);
    const int num_groups = is_constant_B ? 1 : B.size(1);
    const int chunk_size = kernel_config::chunk_size;
    const int num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    at::Tensor du = torch::empty_like(u);
    at::Tensor ddelta = torch::empty_like(delta);
    at::Tensor dA = torch::zeros_like(A);
    at::Tensor dB = is_constant_B 
        ? torch::zeros_like(B) 
        : torch::zeros_like(B, B.options().dtype(torch::kFloat32));
    at::Tensor dC = is_constant_C 
        ? torch::zeros_like(C) 
        : torch::zeros_like(C, C.options().dtype(torch::kFloat32));
    at::Tensor dD;
    if (D_.has_value()) {
        dD = torch::zeros_like(D_.value());
    }
    at::Tensor ddelta_bias;
    if (delta_bias_.has_value()) {
        ddelta_bias = torch::zeros_like(delta_bias_.value());
    }

    at::Tensor out;

    BackwardParams params;
    set_backward_params(
        params,
        batch_size, 
        num_channels, 
        seq_len, 
        state_dim,
        num_groups,
        num_chunks,
        is_constant_B,
        is_constant_C,
        u, 
        delta, 
        A, 
        B, 
        C, 
        out,
        D_.has_value() ? D_.value().data_ptr() : nullptr,
        delta_bias_.has_value() ? D_.value().data_ptr() : nullptr,
        x_.has_value() ? x_.value().data_ptr() : nullptr,
        dout, 
        du, 
        ddelta, 
        dA, 
        dB, 
        dC,
        D_.has_value() ? dD.data_ptr() : nullptr,
        delta_bias_.has_value() ? ddelta_bias.data_ptr() : nullptr,
        use_delta_softplus
    );

    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    backward_kernel_launch(params, stream);

    std::vector<at::Tensor> result = {du, ddelta, dA, dB, dC, dD, ddelta_bias};
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selective_scan_forward, "Selective scan forward");
    m.def("backward", &selective_scan_backward, "Selective scan backward");
}

