import pytest

pytest_plugins = ['pytest_userver.plugins.core', 'pytest_userver.plugins.service']


@pytest.fixture
def userver_config_patch():
    def patch(config_yaml, _config_vars):
        http_cfg = config_yaml['components_manager']['components'].get('http-client')
        if http_cfg:
            http_cfg.pop('fs-task-processor', None)
    return patch
