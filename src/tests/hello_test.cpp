#include <matrix_service/core/greeting.hpp>
#include <userver/utest/utest.hpp>

UTEST(SayHelloTo, Basic) {
    EXPECT_EQ(matrix_service::SayHelloTo("Developer"), "Hello, Developer!\n");
    EXPECT_EQ(matrix_service::SayHelloTo({}), "Hello, unknown user!\n");
}
