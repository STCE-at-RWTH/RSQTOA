#include <cmath>
#include <random>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#ifndef RSQTOA_CASE_STUDIES_INCLUDE_DIFFUSION_HPP_
#define RSQTOA_CASE_STUDIES_INCLUDE_DIFFUSION_HPP_

template <typename T, int N = Eigen::Dynamic>
using VT = Eigen::Matrix<T, N, 1>;

template <typename T>
using MT = Eigen::SparseMatrix<T>;

// Rhs of ode
template <typename T, int N = Eigen::Dynamic>
static inline void g(const VT<T, N> &y, const T &yl, const T &yr, VT<T, N> &r) {

  const int n = y.size();
  const int dt_rec = (n+1)*(n+1);

  r(0) = dt_rec * (yl - 2*y(0) + y(1));
  for (int i = 1; i < n-1; i++) {
    r(i) = dt_rec * (y(i-1) - 2*y(i) + y(i+1));
  }
  r(n-1) = dt_rec * (y(n-2) - 2*y(n-1) + yr);

}

// Tangent of rhs of ode
template <typename T, int N = Eigen::Dynamic>
static inline void g_t(const VT<T, N> &y_t, VT<T, N> &r_t) {

  const int n = y_t.size(), dt_rec = (n+1)*(n+1);
  
  r_t(0) = dt_rec * (-2*y_t(0) + y_t(1));
  for (int i = 1; i < n-1; i++) {
    r_t(i) = dt_rec * (y_t(i-1) - 2*y_t(i) + y_t(i+1));
  }
  r_t(n-1) = dt_rec * (y_t(n-2) - 2*y_t(n-1));

}

// Jacobian of rhs of ode
template <typename T, int N = Eigen::Dynamic>
static inline void dgdy(MT<T> &A) {

  const int n = A.cols();
  VT<T, N> y_t(n), r_t(n);
  std::vector<Eigen::Triplet<T>> entries;
  entries.reserve(2*n-1);

  for (int i = 0; i < n; i++) {
    y_t = VT<T, N>::Unit(n, i);
    g_t(y_t, r_t);
    if (i > 0) {
      entries.push_back(Eigen::Triplet<T>(i, i-1, r_t(i-1)));
    }
    entries.push_back(Eigen::Triplet<T>(i, i, r_t(i)));
  }
  A.setFromTriplets(entries.begin(), entries.end());

}

// Residual of nls
template <typename T, int N = Eigen::Dynamic>
static inline void f(
  const T &delta_t, const VT<T, N> &y, const T &yl, const T &yr,
  const VT<T, N> &y_prev, VT<T, N> &r
) {

  g(y, yl, yr, r);
  r = y - y_prev - r * delta_t;

}

// Jacobian of residual of nls wrt. state
template <typename T, int N = Eigen::Dynamic>
static inline void dfdy(
  const T &delta_t, const T &c, const VT<T, N> &y, MT<T> &A
) {

  const int n = y.size();
  dgdy(A);
  A *= -delta_t * c; 
  MT<T> B(n, n);
  B.setIdentity();
  A += B;

}

// Newton solver for nls
template <typename T, int N = Eigen::Dynamic>
static inline void newton(
  const T &delta_t, const T &c, const VT<T, N> &y_prev,
  const T &yl, const T &yr, VT<T, N> &y
) {

  const int n = y.size();
  const T eps = std::sqrt(std::numeric_limits<T>::epsilon());

  MT<T> A(n, n);
  A.reserve(3*n - 2);
  VT<T, N> r(VT<T, N>::Zero(n));
  f(delta_t, y, yl, yr, y_prev, r);

  Eigen::SimplicialLLT<MT<T>> solver;
  auto rnorm_old = std::numeric_limits<T>::max();
  while (std::abs(r.norm() - rnorm_old) > eps) {
    rnorm_old = r.norm();

    dfdy(delta_t, c, y, A);
    solver.compute(A);
    r = solver.solve(r);
    y -= r;
    f(delta_t, y, yl, yr, y_prev, r);
  }

  //std::cout << "Norm: " << r.norm() << std::endl;

}

// Implicit Euler integration
template <typename T, int N = Eigen::Dynamic>
static inline void euler(
  const int &m, const int &ncs, const T &time, const T &c, 
  const T &yl, const T &yr, VT<T, N> &y
) {

  const T delta_t = time / m;

  for (int j = 0; j < m; j += ncs) {
    for (int i = j; i < std::min(j+ncs, m); i++) {
      VT<T, N> y_prev = y;
      newton(delta_t, c, y_prev, yl, yr, y);
    }
  }

}

static constexpr int32_t diffusion_dims = 3;

//! Type of the input vector
template<typename T>
using input_vector_t = Eigen::Matrix<T, diffusion_dims, 1>;

/******************************************************************************
 * Original model
 ******************************************************************************/
template<typename T>
static inline T diffusion(const input_vector_t<T> &x) {

  // Discretization
  const int n = 101;
  const int ncs = 10;
  const T time = T(x[0]);
  const int m = 10;

  // Heat distribution
  const T c = T(1);
  const T yl = 1;
  const T yr = 0;
  VT<T> y(VT<T>::Ones(n) * x[1]);

  // std::cout << "Temp. at " << x[2] << " from " << x[1] << " after " << x[0] << " => " << std::flush;

  // Call euler
  euler(m, ncs, time, c, yl, yr, y);

  // Interpolation
  const int k = std::floor(x[2] * (n-1));
  const T w = x[2] * (n-1) - k;

  // Right bound edge case‚
  if (k == n-1) {
    // std::cout << y[k] << std::endl;
    return y[k];
  }

  // Return interpolated temperature at given location
  // std::cout << (1 - w) * y[k] + w * y[k+1] << std::endl;
  return (1 - w) * y[k] + w * y[k+1];

}

// Model functor
template<typename T>
struct diffusion_functor {
  T operator() (const input_vector_t<T> &x) const {
    return diffusion(x);
  }
};

#endif  // RSQTOA_CASE_STUDIES_INCLUDE_DIFFUSION_HPP_
