#include <matrix_service/core/greeting.hpp>
#include <matrix_service/routes/hello_handler.hpp>
#include <userver/server/handlers/http_handler_base.hpp>

namespace matrix_service {

namespace {

class Hello final : public userver::server::handlers::HttpHandlerBase {
 public:
    static constexpr std::string_view kName = "handler-hello";

    using HttpHandlerBase::HttpHandlerBase;

    std::string HandleRequestThrow(
        const userver::server::http::HttpRequest& request,
        userver::server::request::RequestContext&) const override {
        return matrix_service::SayHelloTo(request.GetArg("name"));
    }
};

}  // namespace

void AppendHello(userver::components::ComponentList& component_list) {
    component_list.Append<Hello>();
}

}  // namespace matrix_service
