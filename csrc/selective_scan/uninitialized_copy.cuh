#pragma once

#include <cub/config.cuh>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace detail {
    template<typename T, typename U>
    __host__ __device__
    void uninitialized_copy(T* ptr, U&& val) {
        if constexpr (::cuda::std::is_trivially_copyable<T>::value) {
            *ptr = ::cuda::std::forward<U>(val);
        }
        else {
            new (ptr) T(::cuda::std::forward<U>(val));
        }
    }
}