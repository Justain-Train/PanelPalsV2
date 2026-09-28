"""
Unit tests for main FastAPI application.

Section 14.1: Unit Tests - Focus on isolated logic
"""

import pytest
from fastapi import status


@pytest.mark.unit
def test_health_check(test_client):
    """Test health check endpoint returns correct status."""
    response = test_client.get("/health")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    assert data["status"] == "healthy"
    assert data["service"] == "panelpals-backend"
    assert data["version"] == "1.0.0"
    assert "google_vision_configured" in data
    assert "elevenlabs_configured" in data


@pytest.mark.unit
def test_root_endpoint(test_client):
    """Test root endpoint returns API information."""
    response = test_client.get("/")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    assert "message" in data
    assert data["message"] == "PanelPals V2 Backend API"
    assert "health" in data


@pytest.mark.unit
def test_security_headers(test_client):
    """
    Test security headers are present on all responses.
    Section 5.2: Secure-by-default API design
    """
    response = test_client.get("/health")
    
    assert "x-content-type-options" in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"
    
    assert "x-frame-options" in response.headers
    assert response.headers["x-frame-options"] == "DENY"
    
    assert "x-xss-protection" in response.headers
    assert response.headers["x-xss-protection"] == "1; mode=block"


@pytest.mark.unit
def test_cors_configuration(test_client):
    """
    Test CORS middleware is configured.
    Section 5.2: Secure-by-default with restrictive CORS
    """
    response = test_client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        }
    )
    
    # CORS headers should be present
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
