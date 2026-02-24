#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
    [&] {                                       \
        if (COND) {                             \
            constexpr bool CONST_NAME = true;   \
            return __VA_ARGS__();               \
        }                                       \
        else {                                  \
            constexpr bool CONST_NAME = false;  \
            return __VA_ARGS__();               \
        }                                       \
    }()                                         

#define DISPATCH_SWITCH(VALUE, CONST_NAME, ...) \
    [&] {                                       \
        if (VALUE < 16) {                       \
            constexpr int CONST_NAME = 16;      \
            __VA_ARGS__();                      \
        }                                       \
        else if (VALUE < 32) {                  \
            constexpr int CONST_NAME = 32;      \
            __VA_ARGS__();                      \
        }                                       \
        else if (VALUE < 64) {                  \
            constexpr int CONST_NAME = 64;      \
            __VA_ARGS__();                      \
        }                                       \
        else if (VALUE < 128) {                 \
            constexpr int CONST_NAME = 128;     \
            __VA_ARGS__();                      \
        }                                       \
        else {                                  \
            constexpr int CONST_NAME = 256;     \
            __VA_ARGS__();                      \
        }                                       \
    }()                                         