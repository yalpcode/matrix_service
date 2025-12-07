#include <matrix_service/core/greeting.hpp>

#include <fmt/format.h>

namespace service_template {

std::string SayHelloTo(std::string_view name) {
  if (name.empty()) {
    name = "unknown user";
  }

  return fmt::format("Hello, {}!\n", name);
}

}  // namespace service_template
