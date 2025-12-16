#include <matrix_service/schemas/cnn_response_schema.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <userver/formats/json/value_builder.hpp>
#include <userver/formats/serialize/common_containers.hpp>

CnnResponseSchema::CnnResponseSchema(const Tensor<double>& logits) {
    if (logits.dim() != 2) {
        throw std::invalid_argument("logits tensor must be 2D [N, classes]");
    }

    const auto& shape = logits.shape();
    const size_t batch = shape[0];
    const size_t classes = shape[1];

    if (batch != 1) {
        throw std::invalid_argument(
            "Single prediction response expects batch size 1");
    }

    size_t argmax = 0;
    double best = logits({0, 0});
    for (size_t c = 1; c < classes; ++c) {
        const auto val = logits({0, c});
        if (val > best) {
            best = val;
            argmax = c;
        }
    }

    userver::formats::json::ValueBuilder builder;
    builder["prediction"] = static_cast<int>(argmax);
    json_ = builder.ExtractValue();
}

userver::formats::json::Value CnnResponseSchema::GetJson() const {
    return json_;
}
