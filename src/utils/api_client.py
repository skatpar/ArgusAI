"""
API Client for External Model Inference
Handles REST API calls to external model endpoints (e.g., KNIME)
"""

import requests
import base64
from typing import Dict, Any
from datetime import datetime


def build_headers(auth_type: str, api_key: str, content_type: str = "application/json",
                  custom_headers: Dict = None) -> Dict[str, str]:
    """Build request headers based on authentication type"""
    headers = {"Content-Type": content_type}

    if auth_type == "API Key":
        headers["X-API-Key"] = api_key
    elif auth_type == "Bearer Token":
        headers["Authorization"] = f"Bearer {api_key}"
    elif auth_type == "Basic Auth":
        encoded = base64.b64encode(api_key.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"

    if custom_headers:
        headers.update(custom_headers)

    return headers


def call_api_inference(endpoint_url: str, transaction_data: Dict[str, Any],
                       auth_type: str = "None", api_key: str = "",
                       timeout: int = 30, content_type: str = "application/json",
                       custom_headers: Dict = None) -> Dict[str, Any]:
    """
    Call external API for model inference

    Args:
        endpoint_url: Full URL of the API endpoint
        transaction_data: Transaction features as dictionary
        auth_type: Authentication type (None, API Key, Bearer Token, Basic Auth)
        api_key: API key or token
        timeout: Request timeout in seconds
        content_type: Content type header
        custom_headers: Additional custom headers

    Returns:
        Dictionary with success status, response data, and metadata
    """
    headers = build_headers(auth_type, api_key, content_type, custom_headers)

    try:
        start_time = datetime.now()

        response = requests.post(
            endpoint_url,
            json=transaction_data,
            headers=headers,
            timeout=timeout
        )

        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        # Parse response (try JSON first, fallback to text)
        try:
            response_data = response.json()
        except:
            response_data = response.text

        # Handle both success and error responses
        if response.status_code == 200:
            return {
                'success': True,
                'status_code': response.status_code,
                'data': response_data,
                'error': None,
                'latency_ms': latency_ms,
                'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'endpoint': endpoint_url
            }
        else:
            # Error response - check if JSON format
            error_message = response_data if isinstance(response_data, str) else str(response_data)
            return {
                'success': False,
                'status_code': response.status_code,
                'data': None,
                'error': error_message,
                'error_data': response_data if isinstance(response_data, dict) else None,
                'latency_ms': latency_ms,
                'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'endpoint': endpoint_url
            }

    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timed out',
            'status_code': None,
            'data': None,
            'latency_ms': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'endpoint': endpoint_url
        }
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'error': 'Connection failed - could not reach API endpoint',
            'status_code': None,
            'data': None,
            'latency_ms': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'endpoint': endpoint_url
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'status_code': None,
            'data': None,
            'latency_ms': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'endpoint': endpoint_url
        }


def test_api_connection(endpoint_url: str, auth_type: str = "None", api_key: str = "",
                        timeout: int = 30, content_type: str = "application/json",
                        custom_headers: Dict = None) -> Dict[str, Any]:
    """
    Test API connection with a simple request

    Args:
        endpoint_url: Full URL of the API endpoint
        auth_type: Authentication type
        api_key: API key or token
        timeout: Request timeout in seconds
        content_type: Content type header
        custom_headers: Additional custom headers

    Returns:
        Dictionary with connection test results
    """
    test_data = {"test": "connection", "ping": True}

    return call_api_inference(
        endpoint_url=endpoint_url,
        transaction_data=test_data,
        auth_type=auth_type,
        api_key=api_key,
        timeout=timeout,
        content_type=content_type,
        custom_headers=custom_headers
    )
