#ifndef TRAITS_H
#define TRAITS_H

#include "../interface/containers.h"

namespace Wally::Abstract {
    template<typename T>
    struct container_traits {};

    enum class container {
        vector, 
        matrix
    };

    template<typename T>
    struct container_traits<Vector<T>> {
        static constexpr container container_type = container::vector;
        using element_type = T;
    };

    template<typename T>
    struct container_traits<Matrix<T>> {
        static constexpr container container_type = container::matrix;
        using element_type = T;
    };

    template<typename T>
    concept Container = requires(T x) {
        typename container_traits<T>::container_type;
        x.data();
        x.size();
    };
}

#endif