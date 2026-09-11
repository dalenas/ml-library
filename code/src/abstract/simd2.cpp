#include "../../lib/abstract/simd.h"

using namespace Wally;
using namespace Abstract;

__m256 SIMD::SIMD_::_mm256_loadu(const float* const addr) { return _mm256_loadu_ps(addr); }
__m256i SIMD::SIMD_::_mm256_loadu(const int* const addr) { return _mm256_loadu_epi32(addr); }

__m256 SIMD::SIMD_::_mm256_maskloadu(const float* const addr, std::size_t k) { return _mm256_maskload_ps(addr, masks[k]); }
__m256i SIMD::SIMD_::_mm256_maskloadu(const int* const addr, std::size_t k) { return _mm256_maskload_epi32(addr, masks[k]); }

void SIMD::SIMD_::_mm256_storeu(float* addr, __m256 a) { _mm256_storeu_ps(addr, a); }
void SIMD::SIMD_::_mm256_storeu(int* addr, __m256i a) { _mm256_storeu_epi32(addr, a); }

void SIMD::SIMD_::_mm256_maskstoreu(float* addr, __m256 a, std::size_t k) { _mm256_maskstore_ps(addr, masks[k], a); }
void SIMD::SIMD_::_mm256_maskstoreu(int* addr, __m256i a, std::size_t k) { _mm256_maskstore_epi32(addr, masks[k], a); }

__m256 SIMD::SIMD_::_mm256_set1(float k) { return _mm256_set1_ps(k); }
__m256 SIMD::SIMD_::_mm256_set1(int k) { return _mm256_set1_ps(static_cast<float>(k)); }            // may want to change to m256i, which would require if constexpr conditions in scalar operations

template<typename T>
auto SIMD::SIMD_::_mm256_setzero() {
    if constexpr(std::is_same_v<T, float>)
        return _mm256_setzero_ps();
    else
        return _mm256_setzero_epi32();
}

__m256 SIMD::SIMD_::_mm256_add(const __m256 a, const __m256 b) { return _mm256_add_ps(a, b); }
__m256 SIMD::SIMD_::_mm256_add(__m256 a, __m256i b) { 
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_add_ps(a, bf);
}
__m256 SIMD::SIMD_::_mm256_add(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_add_ps(af, b);
}
__m256i SIMD::SIMD_::_mm256_add(__m256i a, __m256i b) { return _mm256_add_epi32(a, b); }

__m256 SIMD::SIMD_::_mm256_sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
__m256 SIMD::SIMD_::_mm256_sub(__m256 a, __m256i b) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_sub_ps(a, bf);
}
__m256 SIMD::SIMD_::_mm256_sub(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_sub_ps(af, b);
}
__m256i SIMD::SIMD_::_mm256_sub(__m256i a, __m256i b) { return _mm256_sub_epi32(a, b); }

__m256 SIMD::SIMD_::_mm256_mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
__m256 SIMD::SIMD_::_mm256_mul(__m256 a, __m256i b) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_mul_ps(a, bf);
}
__m256 SIMD::SIMD_::_mm256_mul(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_mul_ps(af, b);
}
__m256i SIMD::SIMD_::_mm256_mul(__m256i a, __m256i b) { return _mm256_mullo_epi32(a, b); }

__m256 SIMD::SIMD_::_mm256_div(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
__m256 SIMD::SIMD_::_mm256_div(__m256 a, __m256i b) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_div_ps(a, bf);
}
__m256 SIMD::SIMD_::_mm256_div(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_div_ps(af, b);
}
__m256 SIMD::SIMD_::_mm256_div(__m256i a, __m256i b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_div_ps(af, bf);
}

float SIMD::SIMD_::_mm256_sum(__m256 a) {
    __m128 high = _mm256_castps256_ps128(a);
    __m128 low = _mm256_extractf32x4_ps(a, 1);

    __m128 sum_vec = _mm_add_ps(high, low);

    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);

    return _mm_cvtss_f32(sum_vec);
}

int SIMD::SIMD_::_mm256_sum(__m256i a) {
    __m128i high = _mm256_castsi256_si128(a);
    __m128i low = _mm256_extracti32x4_epi32(a, 1);

    __m128i sum_vec = _mm_add_epi32(high, low);

    sum_vec = _mm_hadd_epi32(sum_vec, sum_vec);
    sum_vec = _mm_hadd_epi32(sum_vec, sum_vec);

    return _mm_cvtsi128_si32(sum_vec);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }

__m256 SIMD::SIMD_::_mm256_fmadd(__m256 a, __m256 b, __m256i c) {
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmadd_ps(a, b, cf);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256 a, __m256i b, __m256 c) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_fmadd_ps(a, bf, c);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256 a, __m256i b, __m256i c) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmadd_ps(a, bf, cf);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256i a, __m256 b, __m256 c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_fmadd_ps(af, b, c);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256i a, __m256 b, __m256i c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmadd_ps(af, b, cf);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256i a, __m256i b, __m256 c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_fmadd_ps(af, bf, c);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256i a, __m256i b, __m256i c) { 
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmadd_ps(af, bf, cf);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256 a, __m256 b, __m256 c) { return _mm256_fmsub_ps(a, b, c); }

__m256 SIMD::SIMD_::_mm256_fmsub(__m256 a, __m256 b, __m256i c) {
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmsub_ps(a, b, cf);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256 a, __m256i b, __m256 c) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_fmsub_ps(a, bf, c);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256 a, __m256i b, __m256i c) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmsub_ps(a, bf, cf);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256i a, __m256 b, __m256 c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_fmsub_ps(af, b, c);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256i a, __m256 b, __m256i c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmsub_ps(af, b, cf);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256i a, __m256i b, __m256 c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_fmsub_ps(af, bf, c);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256i a, __m256i b, __m256i c) { 
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmsub_ps(af, bf, cf);
}

__m256 SIMD::SIMD_::_mm256_pow(__m256 a, __m256 b) { return _mm256_pow_ps(a, b); }

__m256 SIMD::SIMD_::_mm256_pow(__m256 a, __m256i b) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_pow_ps(a, bf);
}

__m256 SIMD::SIMD_::_mm256_pow(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_pow_ps(af, b);
}

__m256 SIMD::SIMD_::_mm256_pow(__m256i a, __m256i b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_pow_ps(af, bf);
}

__m256 SIMD::SIMD_::_mm256_abs(__m256 a) {
    const __m256 sign_vec = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign_vec, a);
}

__m256i SIMD::SIMD_::_mm256_abs(__m256i a) {
    return _mm256_abs_epi32(a);
}

__m256 SIMD::SIMD_::_mm256_rcp(__m256 a) {
    return _mm256_rcp_ps(a);
}

__m256 SIMD::SIMD_::_mm256_rcp(__m256i a) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_rcp_ps(af);
}

template<typename ContainerY>
void setzero(ContainerY& Y) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<U, float>) {
        const __m256 zero = _mm256_setzero_ps();

        for(; i < EDGE; i += WIDTH)
            _mm256_storeu_ps(y + i, zero);

        if(REMAINDER != 0)
            SIMD_::store_k(y + i, zero, REMAINDER);
    } 
    else if constexpr(std::is_same_v<U, int>) {
        const __m256i zero = _mm256_setzero_si256();

        for(; i < EDGE; i += WIDTH)
            _mm256_storeu_si256(y + i, zero);

        if(REMAINDER != 0)
            SIMD_::store_k(y + i, zero, REMAINDER);
    }
}

template<typename T>
T sum(const Vector<T>& A) {
    const T* const a = A.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    auto sum_vec = SIMD_::_mm256_setzero<T>();
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);

        sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);

        sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);
    }

    return SIMD_::_mm256_sum(sum_vec);
}

template<typename T>
void sum(const Matrix<T>& A, Vector<T>& Y) {
    const T* const a = A.data();
    T* const y = Y.data();

    const std::size_t N = A.rows();
    const std::size_t M = A.cols();
    
    if(A.axis() == row) {
        const std::size_t REMAINDER = M % WIDTH;
        const std::size_t EDGE = M - REMAINDER;

        for(std::size_t i = 0; i < N; ++i) {
            const T* const a_row = a + i*M;

            std::size_t j = 0;
            for(; j < EDGE; j += 8) {
                const auto a_vec = SIMD_::_mm256_loadu(a_row + j);
                auto sum_vec = SIMD_::_mm256_loadu(y + j);

                sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);

                SIMD_::__mm256_storeu(y + j, sum_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_row + j, REMAINDER);
                auto sum_vec = SIMD_::_mm256_maskloadu(y + j, REMAINDER);
                
                sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);

                SIMD_::_mm256_maskstoreu(y + j, sum_vec, REMAINDER);
            }
        }
    }
    else if(A.axis() == col) {
        const std::size_t REMAINDER = N % WIDTH;
        const std::size_t EDGE = N - REMAINDER;

        for(std::size_t j = 0; j < M; ++j) {
            const T* const a_col = a + j*N;

            std::size_t i = 0;
            auto sum_vec = SIMD_::_mm256_setzero<T>();
            for(; i < EDGE; i += 8) {
                const auto a_vec = SIMD_::_mm256_loadu(a_col + i);
                
                sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_col + i);

                sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);
            }

            y[j] = SIMD_::_mm256_sum(sum_vec);
        }
    }
}

template<typename T>
T dot(const Vector<T>& A, const Vector<T>& B) {
    const T* const a = A.data();
    const T* const b = B.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - WIDTH;

    std::size_t i = 0;
    auto dot_vec = SIMD_::_mm256_setzero<T>();
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto b_vec = SIMD_::_mm256_loadu(b + i);

        dot_vec = SIMD_::_mm256_hmadd(a_vec, b_vec, dot_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);

        dot_vec = SIMD_::_mm256_hmadd(a_vec, b_vec, dot_vec);
    }

    return SIMD_::_mm256_sum(dot_vec)
}

template<typename S, typename ContainerA, typename ContainerY>
void add(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const auto k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto y_vec = SIMD_::_mm256_add(k_vec, a_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto y_vec = SIMD_::_mm256_add(k_vec, a_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename S, typename ContainerA, typename ContainerY>
void sub(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const auto k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto y_vec = SIMD_::_mm256_sub(k_vec, a_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto y_vec = SIMD_::_mm256_sub(k_vec, a_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename ContainerA, typename S, typename ContainerY>
void sub(const ContainerA& A, const S k, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const auto k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto y_vec = SIMD_::_mm256_sub(a_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto y_vec = SIMD_::_mm256_sub(a_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename S, typename ContainerA, typename ContainerY>
void mul(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const auto k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto y_vec = SIMD_::_mm256_mul(k_vec, a_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto y_vec = SIMD_::_mm256_mul(k_vec, a_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename S, typename ContainerA, typename ContainerY>
void div(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const __m256 k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const __m256 y_vec = SIMD_::_mm256_div(k_vec, a_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const __m256 y_vec = SIMD_::_mm256_div(k_vec, a_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename ContainerA, typename S, typename ContainerY>
void div(const ContainerA& A, const S k, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const __m256 k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const __m256 y_vec = SIMD_::_mm256_div(a_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const __m256 y_vec = SIMD_::_mm256_div(a_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename ContainerA, typename ContainerB, typename ContainerY>
void add(const ContainerA& A, const ContainerB& B, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = typename A_traits::element_type;
    using T = typename B_traits::element_type;
    using U = typename Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type){
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto y_vec = SIMD_::_mm256_add(a_vec, b_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto y_vec = SIMD_::_mm256_add(a_vec, b_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    } // need to add vector + matrix, matrix + vector
}

template<typename ContainerA, typename ContainerB, typename ContainerY>
void sub(const ContainerA&, const ContainerB&, ContainerY&) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = typename A_traits::element_type;
    using T = typename B_traits::element_type;
    using U = typename Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto y_vec = SIMD_::_mm256_sub(a_vec, b_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto y_vec = SIMD_::_mm256_sub(a_vec, b_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    } // need to add vector - matrix, matrix - vector
}

template<typename ContainerA, typename ContainerB, typename ContainerY>
void mul(const ContainerA&, const ContainerB&, ContainerY&) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = typename A_traits::element_type;
    using T = typename B_traits::element_type;
    using U = typename Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto y_vec = SIMD_::_mm256_mul(a_vec, b_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto y_vec = SIMD_::_mm256_mul(a_vec, b_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        } // need to add vector * matrix, matrix * vector
    }
}

template<typename ContainerA, typename ContainerB, typename ContainerY>
void div(const ContainerA&, const ContainerB&, ContainerY&) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = typename A_traits::element_type;
    using T = typename B_traits::element_type;
    using U = typename Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto y_vec = SIMD_::_mm256_div(a_vec, b_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto y_vec = SIMD_::_mm256_div(a_vec, b_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    } // need to add matrix / vector, vector / matrix 
}

template<typename ContainerY, typename S>
void add(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto k_vec = SIMD_::_mm256_set1(k);
    for(; i < EDGE; i += 8) {
        auto y_vec = SIMD_::_mm256_loadu(y + i);
        y_vec = SIMD_::_mm256_add(y_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        auto y_vec = SIMD_::_mm256_maskloadu(y + i, REMAINDER);
        y_vec = SIMD_::_mm256_add(y_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename ContainerY, typename S>
void sub(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto k_vec = SIMD_::_mm256_set1(k);
    for(; i < EDGE; i += 8) {
        auto y_vec = SIMD_::_mm256_loadu(y + i);
        y_vec = SIMD_::_mm256_sub(y_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        auto y_vec = SIMD_::_mm256_maskloadu(y + i, REMAINDER);
        y_vec = SIMD_::_mm256_sub(y_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename ContainerY, typename S>
void mul(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto k_vec = SIMD_::_mm256_set1(k);
    for(; i < EDGE; i += 8) {
        auto y_vec = SIMD_::_mm256_loadu(y + i);
        y_vec = SIMD_::_mm256_mul(y_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        auto y_vec = SIMD_::_mm256_maskloadu(y + i, REMAINDER);
        y_vec = SIMD_::_mm256_mul(y_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename ContainerY, typename S>
void div(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto k_vec = SIMD_::_mm256_set1(k);
    for(; i < EDGE; i += 8) {
        auto y_vec = SIMD_::_mm256_loadu(y + i);
        y_vec = SIMD_::_mm256_div(y_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        auto y_vec = SIMD_::_mm256_maskloadu(y + i, REMAINDER);
        y_vec = SIMD_::_mm256_div(y_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename ContainerA, typename ContainerB, typename ContainerC, typename ContainerY>
void fmadd(const ContainerA& A, const ContainerB& B, const ContainerC& C, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using C_traits = container_traits<ContainerC>;
    using Y_traits = container_traits<ContainerY>;

    using S = A_traits::element_type;
    using T = B_traits::element_type;
    using U = C_traits::element_type;
    using V = Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    const U* const c = C.data();
    V* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type && B_traits::container_type == C_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto c_vec = SIMD_::_mm256_loadu(c + i);
            auto y_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, c_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto c_vec = SIMD_::_mm256_maskloadu(c + i, REMAINDER);
            auto y_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, c_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    }
}

template<typename ContainerA, typename ContainerB, typename ContainerC, typename ContainerY>
void fmsub(const ContainerA& A, const ContainerB& B, const ContainerC& C, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using C_traits = container_traits<ContainerC>;
    using Y_traits = container_traits<ContainerY>;

    using S = A_traits::element_type;
    using T = B_traits::element_type;
    using U = C_traits::element_type;
    using V = Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    const U* const c = C.data();
    V* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type && B_traits::container_type == C_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto c_vec = SIMD_::_mm256_loadu(c + i);
            auto y_vec = SIMD_::_mm256_fmsub(a_vec, b_vec, c_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto c_vec = SIMD_::_mm256_maskloadu(c + i, REMAINDER);
            auto y_vec = SIMD_::_mm256_fmsub(a_vec, b_vec, c_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    }
}

template<typename ContainerA, typename ContainerB, typename ContainerY>
void cross(const ContainerA& A, const ContainerB& B, ContainerY& Y) {
    // this will be assuming A is row-major and B is col-major, A is row vector and B is col-major, or A is row major and B is col vector
    // could implement reverse, but would prob be slower because of load and store latency
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = A_traits::element_type;
    using T = B_traits::element_type;
    using U = Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    if constexpr(A_traits::container_type == matrix && B_traits::container_type == matrix) {
        const std::size_t N = A.rows();
        const std::size_t K = A.cols();
        const std::size_t M = B.cols();
        
        const std::size_t REMAINDER = K % WIDTH;
        const std::size_t EDGE = K - REMAINDER;

        for(std::size_t i = 0; i < N; ++i) {
            const S* const a_row = a + i*K;

            for(std::size_t j = 0; j < M; ++j) {
                const T* const b_col = b + j*K;

                std::size_t k = 0;
                auto sum_vec = SIMD_::_mm256_setzero<U>();
                for(; k < EDGE; k += 8) {
                    const auto a_vec = SIMD_::_mm256_loadu(a_row + k);
                    const auto b_vec = SIMD_::_mm256_loadu(b_col + k);

                    sum_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, sum_vec);
                }

                if(REMAINDER != 0) {
                    const auto a_vec = SIMD_::_mm256_maskloadu(a_row + k, REMAINDER);
                    const auto b_vec = SIMD_::_mm256_maskloadu(b_col + k, REMAINDER);

                    sum_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, sum_vec);
                }

                y[i*M + j] = SIMD_::_mm256_sum(sum_vec);
            }
        }
    }
    else if constexpr(A_traits::container_type == vector || B_traits::container_type == vector) {
        const std::size_t N = Y.size();


    }
}

template<typename T>
float sqsum(const Vector<T>& A) {
    const T* const a = A.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    const __m256 two_vec = SIMD_::_mm256_set1(2.0f);
    for(; i < EDGE; i += WIDTH) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const __m256 pow_vec = SIMD_::_mm256_pow(a_vec, two_vec);

        sum_vec = SIMD_::_mm256_add(sum_vec, pow_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const __m256 pow_vec = SIMD_::_mm256_pow(a_vec, two_vec);

        sum_vec = SIMD_::_mm256_add(sum_vec, pow_vec);
    }

    return SIMD_::_mm256_sum(sum_vec);
}

template<typename T>
void sqsum(const Matrix<T>& A, Vector<T>& Y) {
    const T* const a = A.data();
    T* const y = Y.data();

    const std::size_t N = A.rows();
    const std::size_t M = A.cols();
    
    if(A.axis() == row) {
        const std::size_t REMAINDER = M % WIDTH;
        const std::size_t EDGE = M - REMAINDER;

        for(std::size_t i = 0; i < N; ++i) {
            const T* const a_row = a + i*M;

            std::size_t j = 0;
            for(; j < EDGE; j += WIDTH) {
                const auto a_vec = SIMD_::_mm256_loadu(a_row + j);
                __m256 sum_vec = SIMD_::_mm256_loadu(y + j);

                const __m256 pow_vec = SIMD_::_mm256_mul(a_vec, a_vec);
                sum_vec = SIMD_::_mm256_add(sum_vec, pow_vec);

                SIMD_::_mm256_storeu(y + j, sum_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_row + j, REMAINDER);
                __m256 sum_vec = SIMD_::_mm256_maskloadu(y + j, REMAINDER);

                const __m256 pow_vec = SIMD_::_mm256_mul(a_vec, a_vec);
                sum_vec = SIMD_::_mm256_add(sum_vec, pow_vec);

                SIMD_::_mm256_maskstoreu(y + j, sum_vec, REMAINDER);
            }
        }
    }
    else if(A.axis() == col) {
        const std::size_t REMAINDER = N % WIDTH;
        const std::size_t EDGE = N - REMAINDER;

        for(std::size_t j = 0; j < M; ++j) {
            const T* const a_col = a + j*N;

            std::size_t i = 0;
            __m256 sum_vec = __m256_setzero_ps();
            for(; i < EDGE; i += WIDTH) {
                const auto a_vec = SIMD_::_mm256_loadu(a_col + i);
                const __m256 pow_vec = SIMD_::_mm256_mul(a_vec, a_vec);

                sum_vec = SIMD_::_mm256_add(sum_vec, pow_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_col + i, REMAINDER);
                const __m256 pow_vec = SIMD_::_mm256_mul(a_vec, a_vec);

                sum_vec = SIMD_::_mm256_add(sum_vec, pow_vec);
            }

            y[j] = SIMD_::_mm256_sum(sum_vec);
        }
    }
}

template<typename T>
float lpnorm(const float p, const Vector<T>& A) {
    const T* const a = A.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const __m256 p_vec = SIMD_::_mm256_set1(p);
    __m256 sum_vec = SIMD_::_mm256_setzero<float>();
    for(; i < EDGE; i += 8) {
        auto a_vec = SIMD_::_mm256_loadu(a + i);

        a_vec = SIMD_::_mm256_abs(a_vec);
        const __m256 pow_vec = SIMD_::_mm256_pow(a_vec, p_vec);
        sum_vec = SIMD_::_mm256_add(pow_vec, sum_vec);
    }

    if(REMAINDER != 0) {
        auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);

        a_vec = SIMD_::_mm256_abs(a_vec);
        const __m256 pow_vec = SIMD_::_mm256_pow(a_vec, p_vec);
        sum_vec = SIMD_::_mm256_add(pow_vec, sum_vec);
    }

    float sum = SIMD_::_mm256_sum(sum_vec);
    sum_vec = SIMD_::_mm256_set1(sum);

    __m256 rcp_vec = SIMD_::_mm256_rcp(p_vec);
    __m256 norm_vec = SIMD_::_mm256_pow(sum_vec, )
    return 
}

template<typename S, typename T>
void lpnorm(const S, const Matrix<T>&, Vector<float>&);