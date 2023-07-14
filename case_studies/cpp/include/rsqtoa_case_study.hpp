#include <string>
#include <filesystem>
#include <stdexcept>
#include <fstream>

#ifndef RSQTOA_CASE_STUDIES_RSQTOA_CASE_STUDY_HPP_
#define RSQTOA_CASE_STUDIES_RSQTOA_CASE_STUDY_HPP_

#ifndef RSQTOA_OUTPUT_FOLDER
  #error 'RSQTOA_OUTPUT_FOLDER' not defined.
#endif

namespace rsqtoa {

/******************************************************************************
 * Read epsilon file and return the stored epsilon
 ******************************************************************************/
static inline double read_epsilon(const std::filesystem::path &epsilon_file) {

  if (!std::filesystem::exists(epsilon_file)) {
    std::string error_message = "RSQTOA - Epsilon file not found at '";
    error_message += epsilon_file.c_str();
    error_message += "'!";
    throw std::runtime_error(error_message);
  }
  
  // Read first line of epsilon file
  auto stream = std::ifstream(epsilon_file);
  double epsilon;
  stream >> epsilon;

  return epsilon;

}

} // namespace rsqtoa

#endif  // RSQTOA_CASE_STUDIES_RSQTOA_CASE_STUDY_HPP_
