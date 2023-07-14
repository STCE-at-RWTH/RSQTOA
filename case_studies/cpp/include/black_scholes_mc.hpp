#include <cmath>
#include <random>
#include <Eigen/Dense>

#ifndef RSQTOA_CASE_STUDIES_INCLUDE_BLACK_SCHOLES_MC_HPP_
#define RSQTOA_CASE_STUDIES_INCLUDE_BLACK_SCHOLES_MC_HPP_

static constexpr int32_t f_dims = 4;

//! Type of the input vector
template<typename T>
using input_vector_t = Eigen::Matrix<T, f_dims, 1>;

template<typename T>
static std::vector<T> get_MC_samples() {
  static std::vector<T> dW;
  if (dW.empty()) {
    // Amount of MC samples
    constexpr int32_t np = 1000; 
    dW = std::vector<T>(np);

    std::mt19937 generator;
    generator.seed(4711);
    std::normal_distribution<T> distribution(0.0, 1.0);
    for (int32_t i = 0; i < np; i++) {
      dW[i] = distribution(generator);
    }
  }

  return dW;
}

/******************************************************************************
 * Original model
 ******************************************************************************/
template<typename T>
static inline T f(const input_vector_t<T> &x) {
  const T s0 = x[0];
  const T e = x[1];
  const T r = x[2];
  const T sigma = x[3];

  auto dW = get_MC_samples<float>();

  T p = 0;
  for (size_t i = 0; i < dW.size(); i++) {
    auto s = s0 * std::exp((r - 0.5*sigma*sigma) + sigma*dW[i]);
    p = p + std::exp(-r) * std::max(s-e, 0.0);
  }
  p /= dW.size();

  return p;
}

// Model functor
template<typename T>
struct f_functor {
  T operator() (const input_vector_t<T> &x) const {
    return f(x);
  }
};

#endif  // RSQTOA_CASE_STUDIES_INCLUDE_BLACK_SCHOLES_MC_HPP_
