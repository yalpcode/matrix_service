#include <fmt/format.h>
#include <matrix_service/core/greeting.hpp>

namespace matrix_service {

std::string SayHelloTo(std::string_view name) {
    if (name.empty()) {
        name = "unknown user";
    }

    return fmt::format("Hello, {}!\n", name);
}

}  // namespace matrix_service
