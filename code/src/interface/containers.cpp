#include "../../lib/interface/containers.h"

using namespace Wally;

template<typename T>
Vector<T>::Vector(std::size_t s)
    : size_(s) { data_.reserve(size_); }

template<typename T>
Vector<T>::Vector(std::size_t s, T init)
    : size_(s) { data_.assign(size_, init); }

template<typename T>
T& Vector<T>::operator()(std::size_t i) {
    return data_[i];
}

template<typename T>
const T& Vector<T>::operator()(std::size_t) const {
    return data_[i];
}

template<typename T>
const std::size_t Vector<T>::size() const {
    return size_;
}

template<typename T>
T* Vector<T>::data() {
    return data_.data();
}

template<typename T>
const T* Vector<T>::data() const {
    return data_.data();
}

template<typename U>
Matrix<U>::Matrix(std::size_t r, std::size_t c, Major a)
    : rows_(r), cols_(c), axis_(a) { data_.reserve(rows_ * cols_); }

template<typename U>
Matrix<U>::Matrix(std::size_t r, std::size_t c, Major a, U init)
    : rows_(r), cols_(c), axis_(a) { data_.assign(rows_* cols_, init); }

template<typename U>
U& Matrix<U>::operator()(std::size_t row, std::size_t col) {
    if(axis_ == col)
        return data_[row + col*rows_];
    return data_[col + row*cols_];
}

template<typename U>
const U& Matrix<U>::operator()(std::size_t row, std::size_t col) const {
    if(axis_ == col)
        return data_[row + col*rows_];
    return data_[col + row*cols_];
}

template<typename U>
const std::size_t Matrix<U>::rows() const {
    return rows_;
}

template<typename U>
const std::size_t Matrix<U>::cols() const {
    return cols_;
}

template<typename U>
const std::size_t Matrix<U>::size() const {
    return rows_ * cols_;
}

template<typename U>
const Major Matrix<U>::axis() const {
    return axis_;
}


template<typename U>
U* Matrix<U>::data() {
    return data_.data();
}

template<typename U>
const U* Matrix<U>::data() const {
    return data_.data();
}