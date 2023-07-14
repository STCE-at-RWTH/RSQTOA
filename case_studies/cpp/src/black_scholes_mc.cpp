#include <filesystem>

#include <Eigen/Dense>
#include <dco.hpp>

#include "rsqtoa_case_study.hpp"

// Approximator (ML)
#include "approximator/rsqtoa_approximator.hpp"
using namespace rsqtoa;

// Original model
#include "black_scholes_mc.hpp"

template<typename T>
static constexpr boundaries_t<T, f_dims>
f_ranges = {{{50, 60}, {52, 62}, {0.04, 0.05}, {0.01, 0.02}}};

#ifndef RSQTOA_LEARN_ORIGINAL_MODEL
  // Generated regression model
  #include "black_scholes_mc_rsqtoa.hpp"

  // Use original model
  #define FULL_FUNCTOR f_functor
  #define FULL_DIMS f_dims
  #define FULL_RANGES f_ranges

  // Use reduced model
  #define RSQTOA_FUNCTOR f_reduced_functor
  #define RSQTOA_DIMS f_reduced_dims
  #define RSQTOA_RANGES f_reduced_ranges
#else
  // Use original model
  #define RSQTOA_FUNCTOR f_functor
  #define RSQTOA_DIMS f_dims
  #define RSQTOA_RANGES f_ranges
#endif

// Entry point
int main(const int, const char **) {

  // Config
  using base_t = float;
  const int32_t epochs = 5;
  const int32_t samples = std::pow(15, f_dims);

  // Setup the approximator
  Approximator<base_t, RSQTOA_FUNCTOR, RSQTOA_DIMS> approx;
  approx.init_dataset(RSQTOA_RANGES<base_t>, samples);
  approx.set_epochs(epochs);
  approx.set_output_folder(RSQTOA_OUTPUT_FOLDER);

  // Read and set epsilon 
  #ifndef RSQTOA_LEARN_ORIGINAL_MODEL
  const base_t eps = read_epsilon(RSQTOA_EPSILON_FILE);
  approx.set_epsilon(eps);
  #endif

  // Train the model
  Model<base_t, RSQTOA_DIMS> m = approx.train_model();

  #ifndef RSQTOA_LEARN_ORIGINAL_MODEL
  Dataset_plain<base_t, FULL_FUNCTOR, FULL_DIMS> test_data(FULL_RANGES<base_t>);
  test_data.sample_domain(300);

  // Approximated reduced model
  auto approximated_reduced_model = [&m](const f_reduced_input_t<base_t> &x) {
    return m.predict_single(x);
  };

  // Reassamble the model
  auto f_approx = [&approximated_reduced_model](
    const f_full_input_t<base_t> &x
  ) {
    return f_reassambled_model(x, approximated_reduced_model);
  };

  // Evaluate models and calculate mean squared loss
  const base_t error = test_data.check_model(f_approx);
  
  if (error <= eps) {
    std::cout << "SUCCESS: " << error << " <= " << eps << " (eps)!\n";
  } else {
    std::cout << "FAILURE: " << error << " > " << eps << " (eps)!\n";
  }
  #endif

}
