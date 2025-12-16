**This repository is for userver v2.8 or older versions. For newer versions of userver please use 
[userver-create-service](https://userver.tech/de/dab/md_en_2userver_2build_2build.html#autotoc_md177) script.**

# Matrix Service

Сервис на userver, предоставляет перемножение матриц и прямой проход через простую CNN для mnist.

## Endpoints

- `GET /ping` — проверка healthy.
- `GET /matrix-mul` — умножение матриц; принимает JSON-тело с `left` и `right` (см. `openapi.yaml`).
- `GET /simple-cnn` — прогон SimpleCNN, на выходе `prediction` (индекс класса после softmax). Параметры запроса (query):
  - `inputs` (обязательно): строка с JSON массивом формы `[batch, height, width]`; допустим только batch=1, изображения 28x28 (grayscale).
  - `weights` (опционально): строка с JSON весами `conv1/fc1/fc2` как в `openapi.yaml`; если нет — используются встроенные веса с фиксированным сидом.
  Тело запроса не используется, данные передаются в query. Максимальный размер запроса — 20 МБ.

## Build/Test

Make-пресеты:
- `make cmake-debug` / `make cmake-release`
- `make build-debug` / `make build-release`
- `make test-debug` / `make test-release`

Спецификация OpenAPI лежит в `openapi.yaml`. В `docker-compose` есть Swagger UI на `http://localhost:8081`, он монтирует эту схему.

Для локальной работы предпочтительно `docker compose up` (поднимет сервис и Swagger). Формируйте свои запросы по схемам из `openapi.yaml`, передавая данные в query-параметры.


## License

The original template is distributed under the [Apache-2.0 License](https://github.com/userver-framework/userver/blob/develop/LICENSE)
and [CLA](https://github.com/userver-framework/userver/blob/develop/CONTRIBUTING.md). Services based on the template may change
the license and CLA.
