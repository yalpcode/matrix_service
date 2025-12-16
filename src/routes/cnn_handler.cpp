#include <matrix_service/dtos/cnn_request_dto.hpp>
#include <matrix_service/routes/cnn_handler.hpp>
#include <matrix_service/schemas/cnn_response_schema.hpp>
#include <matrix_service/services/simple_cnn_service.hpp>
#include <userver/formats/json.hpp>
#include <userver/http/common_headers.hpp>
#include <userver/http/content_type.hpp>
#include <userver/server/handlers/http_handler_base.hpp>
#include <userver/server/http/http_method.hpp>
#include <userver/utils/text_light.hpp>

namespace matrix_service {

namespace {

class CnnHandler final : public userver::server::handlers::HttpHandlerBase {
 public:
    static constexpr std::string_view kName = "handler-simple-cnn";

    using HttpHandlerBase::HttpHandlerBase;

    std::string HandleRequestThrow(
        const userver::server::http::HttpRequest& request,
        userver::server::request::RequestContext&) const override {
        auto& http_response = request.GetHttpResponse();
        http_response.SetHeader(std::string_view{"Access-Control-Allow-Origin"},
                                std::string{"*"});
        http_response.SetHeader(
            std::string_view{"Access-Control-Allow-Headers"},
            std::string{"Content-Type"});
        http_response.SetHeader(
            std::string_view{"Access-Control-Allow-Methods"},
            std::string{"POST, OPTIONS"});

        if (request.GetMethod() == userver::server::http::HttpMethod::kOptions) {
            http_response.SetStatus(userver::server::http::HttpStatus::kNoContent);
            return {};
        }

        userver::formats::json::Value payload;

        const auto content_type_header =
            request.GetHeader(userver::http::headers::kContentType);
        if (userver::utils::text::StartsWith(content_type_header,
                                             "multipart/")) {
            if (!request.HasFormDataArg("inputs")) {
                throw std::invalid_argument("Missing form-data part 'inputs'");
            }
            const auto& inputs_arg = request.GetFormDataArg("inputs");
            auto inputs_json =
                userver::formats::json::FromString(inputs_arg.value);

            userver::formats::json::ValueBuilder builder;
            builder["inputs"] = inputs_json;

            if (request.HasFormDataArg("weights")) {
                const auto& weights_arg = request.GetFormDataArg("weights");
                auto weights_json =
                    userver::formats::json::FromString(weights_arg.value);
                builder["weights"] = weights_json;
            }

            payload = builder.ExtractValue();
        } else {
            const auto body = request.RequestBody();
            payload = userver::formats::json::FromString(body);
        }

        CnnRequestDTO dto(payload);
        SimpleCNNService service;
        if (dto.HasWeights()) {
            service.ApplyWeights(dto.GetWeights());
        }
        const auto logits = service.Infer(dto.GetInput());

        CnnResponseSchema schema(logits);
        const auto response_json = schema.GetJson();

        request.GetHttpResponse().SetContentType(
            userver::http::content_type::kApplicationJson);
        return userver::formats::json::ToString(response_json);
    }
};

}  // namespace

void AppendCnn(userver::components::ComponentList& component_list) {
    component_list.Append<CnnHandler>();
}

}  // namespace matrix_service
