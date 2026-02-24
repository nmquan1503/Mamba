#pragma once

namespace kernel_config {
    constexpr int num_threads = 32;
    constexpr int num_elements = 4;
    constexpr int chunk_size = num_threads * num_elements;
}