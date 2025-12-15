FROM ghcr.io/userver-framework/ubuntu-22.04-userver-pg:v2.8

ARG APP_USER=app
ARG APP_HOME=/matrix_service

RUN groupadd -r "${APP_USER}" \
    && useradd -r -g "${APP_USER}" "${APP_USER}" \
    && mkdir -p "${APP_HOME}" \
    && chown -R "${APP_USER}:${APP_USER}" "${APP_HOME}"

WORKDIR ${APP_HOME}
USER ${APP_USER}

CMD ["make", "cmake-debug", "build-debug"]
