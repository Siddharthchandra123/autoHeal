"""Phase 6B viewer: Swagger docs opt-in and secure defaults."""
import pytest
from fastapi.testclient import TestClient
from autotune.viewer_api import create_app


def test_docs_disabled_by_default(monkeypatch):
    monkeypatch.delenv('AUTOTUNE_ENABLE_DOCS', raising=False)
    client = TestClient(create_app())
    assert client.get('/docs').status_code == 404
    assert client.get('/openapi.json').status_code == 404
    assert client.get('/redoc').status_code == 404
    assert client.get('/api/health').status_code == 200


def test_docs_env_opt_in(monkeypatch):
    monkeypatch.setenv('AUTOTUNE_ENABLE_DOCS', '1')
    client = TestClient(create_app())
    assert client.get('/docs').status_code == 200
    schema = client.get('/openapi.json')
    assert schema.status_code == 200
    assert '/api/experiments' in schema.json()['paths']
    assert client.get('/redoc').status_code == 404


def test_docs_enabled_explicitly_with_env_off(monkeypatch):
    monkeypatch.setenv('AUTOTUNE_ENABLE_DOCS', '0')
    client = TestClient(create_app(enable_docs=True))
    assert client.get('/docs').status_code == 200
    assert client.get('/api/health').json()['mode'] == 'read-only'
    assert client.post('/api/experiments').status_code == 405


def test_docs_explicit_disable_overrides_env(monkeypatch):
    monkeypatch.setenv('AUTOTUNE_ENABLE_DOCS', '1')
    client = TestClient(create_app(enable_docs=False))
    assert client.get('/docs').status_code == 404
    assert client.get('/openapi.json').status_code == 404


def test_untrusted_host_still_rejected_with_docs_enabled():
    client = TestClient(create_app(enable_docs=True))
    assert client.get('/docs', headers={'Host': 'external.example'}).status_code == 400
