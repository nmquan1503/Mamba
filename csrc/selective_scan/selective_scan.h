#pragma once

#include <cstdint>

using index_t = uint32_t;

struct ScanParams {
    int batch_size;
    int seq_len;
    int num_chunks;

    index_t A_batch_stride;
    index_t B_batch_stride;
    index_t out_batch_stride;

    void* __restrict A_ptr;
    void* __restrict B_ptr;
    void* __restrict state_ptr;
    void* __restrict out_ptr;
};

struct ForwardParams {
    /*

    u: [batch, channel, seq]
    delta: [batch, channel, seq]
    A: [channel, state]
    B: [batch, group, state, seq_len] || [channel, state]
    C: [batch, group, state, seq]
    D: [channel]
    delta_bias: [channel]
    x: [batch, channel, chunk, state * 2]
    out: [batch, channel, seq]

    */
    int batch_size;
    int seq_len;
    int state_dim;
    int num_channels;
    int num_groups;
    int num_chunks;
    int num_channels_per_group;
    bool is_constant_B;
    bool is_constant_C;
    bool use_delta_softplus;

    index_t A_channel_stride;
    index_t A_state_stride;

    index_t B_batch_stride;
    index_t B_group_stride;
    index_t B_channel_stride;
    index_t B_state_stride;
    
    index_t C_batch_stride;
    index_t C_group_stride;
    index_t C_channel_stride;
    index_t C_state_stride;

    index_t u_batch_stride;
    index_t u_channel_stride;

    index_t delta_batch_stride;
    index_t delta_channel_stride;

    index_t out_batch_stride;
    index_t out_channel_stride;

    void* __restrict A_ptr;
    void* __restrict B_ptr;
    void* __restrict C_ptr;
    void* __restrict D_ptr;
    void* __restrict u_ptr;
    void* __restrict delta_ptr;
    void* __restrict delta_bias_ptr;
    void* __restrict x_ptr;
    void* __restrict out_ptr;
};

struct BackwardParams : public ForwardParams {
    /*
    dout: [batch, channel, seq]
    dA: [channel, state]
    dB: [batch, group, state, seq] || [channel, state] 
    dC: [batch, group, state, seq] || [channel, state] 
    du: [batch, channel, seq]
    ddelta: [batch, channel, seq]
    */
    
    index_t dout_batch_stride;
    index_t dout_channel_stride;
    
    index_t dA_channel_stride;
    index_t dA_state_stride;

    index_t dB_batch_stride;
    index_t dB_group_stride;
    index_t dB_channel_stride;
    index_t dB_state_stride;

    index_t dC_batch_stride;
    index_t dC_group_stride;
    index_t dC_channel_stride;
    index_t dC_state_stride;

    index_t du_batch_stride;
    index_t du_channel_stride;

    index_t ddelta_batch_stride;
    index_t ddelta_channel_stride;

    void* __restrict dout_ptr;
    void* __restrict dA_ptr;
    void* __restrict dB_ptr;
    void* __restrict dC_ptr;
    void* __restrict dD_ptr;
    void* __restrict du_ptr;
    void* __restrict ddelta_ptr;
    void* __restrict ddelta_bias_ptr;
};