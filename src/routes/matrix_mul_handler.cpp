#include <matrix_service/dtos/matrixs_request_dto.hpp>
#include <matrix_service/routes/matrix_mul_handler.hpp>
#include <matrix_service/schemas/matrix_response_schema.hpp>
#include <matrix_service/services/matrix_mul_service.hpp>
#include <userver/formats/json.hpp>
#include <userver/http/content_type.hpp>
#include <userver/server/handlers/http_handler_json_base.hpp>

namespace matrix_service {

namespace {

class MatrixMulHandler final
    : public userver::server::handlers::HttpHandlerBase {
 public:
    static constexpr std::string_view kName = "handler-matrix-mul";

    using HttpHandlerBase::HttpHandlerBase;

    std::string HandleRequestThrow(
        const userver::server::http::HttpRequest& request,
        userver::server::request::RequestContext&) const override {
        const auto body = request.RequestBody();
        auto json = userver::formats::json::FromString(body);

        MatrixsRequestDTO dto(json);
        MatrixMulService service;
        const auto result = service.matrixMul(dto);

        MatrixResponseSchema schema(result);
        const auto response_json = schema.getJson();
        request.GetHttpResponse().SetContentType(
            userver::http::content_type::kApplicationJson);
        return userver::formats::json::ToString(response_json);
    }
};

}  // namespace

void AppendMatrixMul(userver::components::ComponentList& component_list) {
    component_list.Append<MatrixMulHandler>();
}

}  // namespace matrix_service
