#include <cmath>
#include <chrono>
#include <thread>
#include <Eigen/Dense>

#ifndef RSQTOA_CASE_STUDIES_INCLUDE_SHOW_CASE_HPP_
#define RSQTOA_CASE_STUDIES_INCLUDE_SHOW_CASE_HPP_

// Runtime of the model
static constexpr int32_t additional_runtime = 10;

// Input dimensions of the model
static constexpr int32_t f_dims = 2;

//! Type of the input vector
template<typename T>
using input_vector_t = Eigen::Matrix<T, f_dims, 1>;
 
/******************************************************************************
 * Original model
 ******************************************************************************/
template<typename T>
static inline T f(const input_vector_t<T> &x) {

  // Artificially increase runtime for the model
  //std::this_thread::sleep_for(std::chrono::milliseconds(additional_runtime));

  return (
    std::sin(3.14 * x[0]) + std::tan(x[1]) / T(30.0) + 42
  );

}

// Model functor
template<typename T>
struct f_functor {
  T operator() (const input_vector_t<T> &x) const {
    return f(x);
  }
};

#endif  // RSQTOA_CASE_STUDIES_INCLUDE_SHOW_CASE_HPP_
