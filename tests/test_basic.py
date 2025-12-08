# Start via `make test-debug` or `make test-release`
async def test_matrix_mul(service_client):
    response = await service_client.get(
        '/matrix-mul',
        json={'left': [[1, 2], [3, 4]], 'right': [[5], [6]]},
    )
    assert response.status == 200
    assert response.json() == {'result': [[17], [39]]}
