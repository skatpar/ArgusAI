# API Integration Guide

## Overview

The ArgusAI dashboard now includes a REST API client for integrating with external model endpoints, such as KNIME model servers. This allows you to send transaction features as JSON and receive fraud detection scores from your deployed models.

## Features

- ✅ Multiple authentication methods (API Key, Bearer Token, Basic Auth)
- ✅ Custom headers support
- ✅ Request/response logging
- ✅ Automatic error handling and timeout management
- ✅ Latency tracking
- ✅ Compatible with KNIME and other REST APIs

## Quick Start

### 1. Import the API Client

```python
from src.utils.api_client import call_api_inference, test_api_connection
```

### 2. Configure Your API

```python
# Example: KNIME API configuration
endpoint_url = "https://your-knime-server.com/api/v1/predict"
auth_type = "API Key"  # or "Bearer Token", "Basic Auth", "None"
api_key = "your-api-key-here"
timeout = 30  # seconds
```

### 3. Send Transaction for Scoring

```python
# Prepare transaction data
transaction_data = {
    "transaction_amount": 1250.00,
    "merchant_category": "online",
    "transaction_hour": 14,
    "customer_age": 35,
    "account_age_days": 730,
    "previous_transactions": 125,
    "is_foreign": False,
    "distance_from_home": 5.2,
    "device_type": "mobile"
}

# Call API
result = call_api_inference(
    endpoint_url=endpoint_url,
    transaction_data=transaction_data,
    auth_type=auth_type,
    api_key=api_key,
    timeout=timeout
)

# Check result
if result['success']:
    fraud_score = result['data'].get('fraud_score')
    prediction = result['data'].get('prediction')
    latency = result['latency_ms']

    print(f"Fraud Score: {fraud_score}")
    print(f"Prediction: {prediction}")
    print(f"Latency: {latency}ms")
else:
    print(f"Error: {result['error']}")
```

## Authentication Types

### None
No authentication required.

```python
result = call_api_inference(
    endpoint_url="http://localhost:8000/predict",
    transaction_data=data,
    auth_type="None"
)
```

### API Key
Sends API key in `X-API-Key` header.

```python
result = call_api_inference(
    endpoint_url="https://api.example.com/predict",
    transaction_data=data,
    auth_type="API Key",
    api_key="your-api-key-12345"
)
```

### Bearer Token
Sends token in `Authorization: Bearer {token}` header.

```python
result = call_api_inference(
    endpoint_url="https://api.example.com/predict",
    transaction_data=data,
    auth_type="Bearer Token",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
)
```

### Basic Auth
Sends username:password encoded in `Authorization: Basic {encoded}` header.

```python
result = call_api_inference(
    endpoint_url="https://api.example.com/predict",
    transaction_data=data,
    auth_type="Basic Auth",
    api_key="username:password"
)
```

## Custom Headers

Add any additional headers your API requires:

```python
custom_headers = {
    "X-Custom-Header": "value",
    "X-Request-ID": "12345",
    "X-Client-Version": "1.0"
}

result = call_api_inference(
    endpoint_url=endpoint_url,
    transaction_data=data,
    auth_type="API Key",
    api_key="your-key",
    custom_headers=custom_headers
)
```

## Testing API Connection

Before sending production data, test your API connection:

```python
from src.utils.api_client import test_api_connection

test_result = test_api_connection(
    endpoint_url="https://your-knime-server.com/api/predict",
    auth_type="API Key",
    api_key="your-key",
    timeout=30
)

if test_result['success']:
    print("✅ API connection successful!")
    print(f"Latency: {test_result['latency_ms']}ms")
else:
    print(f"❌ Connection failed: {test_result['error']}")
```

## Response Format

The API client returns a dictionary with the following structure:

```python
{
    'success': True,              # Boolean: True if status code 200
    'status_code': 200,           # HTTP status code
    'data': {                     # Response data (if successful)
        'fraud_score': 0.75,
        'prediction': 'fraud',
        'confidence': 0.85,
        'model_version': 'v2.1'
    },
    'error': None,                # Error message (if failed)
    'latency_ms': 45.23,         # Request latency in milliseconds
    'timestamp': '2024-12-11 10:30:45',
    'endpoint': 'https://...'     # Endpoint URL
}
```

## Error Handling

The client handles common errors gracefully:

### Timeout
```python
{
    'success': False,
    'error': 'Request timed out',
    'status_code': None,
    'data': None
}
```

### Connection Error
```python
{
    'success': False,
    'error': 'Connection failed - could not reach API endpoint',
    'status_code': None,
    'data': None
}
```

### HTTP Errors
```python
{
    'success': False,
    'status_code': 404,  # or 500, 401, etc.
    'error': 'Error message from server',
    'data': None
}
```

## KNIME Integration Example

If you have a KNIME Analytics Platform model deployed via KNIME Server:

```python
# KNIME configuration
knime_config = {
    'endpoint_url': 'https://knime-server.company.com:8080/knime/rest/v4/repository/myworkflow:execute',
    'auth_type': 'Basic Auth',
    'api_key': 'knime_user:knime_password',
    'timeout': 60
}

# Send transaction
transaction = {
    "amount": 5000.00,
    "merchant": "crypto_exchange",
    "hour": 23,
    "customer_age": 28,
    "account_days": 45
}

result = call_api_inference(
    endpoint_url=knime_config['endpoint_url'],
    transaction_data=transaction,
    auth_type=knime_config['auth_type'],
    api_key=knime_config['api_key'],
    timeout=knime_config['timeout']
)

if result['success']:
    print(f"KNIME Model Score: {result['data']}")
```

## Best Practices

1. **Store Credentials Securely**: Never hardcode API keys. Use environment variables or secure vaults.

```python
import os
api_key = os.environ.get('KNIME_API_KEY')
```

2. **Set Appropriate Timeouts**: Model inference can take time. Set realistic timeouts.

```python
# For real-time inference
timeout = 30

# For batch processing
timeout = 120
```

3. **Handle Errors Gracefully**: Always check the `success` field before using the response.

```python
if result['success']:
    # Process response
    process_prediction(result['data'])
else:
    # Log error and use fallback
    logging.error(f"API Error: {result['error']}")
    use_fallback_model()
```

4. **Monitor Latency**: Track API latency to identify performance issues.

```python
if result['latency_ms'] > 1000:
    logging.warning(f"High latency: {result['latency_ms']}ms")
```

5. **Log Requests**: Keep request/response logs for debugging and audit.

```python
logging.info(f"API Request: {transaction_data}")
logging.info(f"API Response: {result}")
```

## Integration with Streamlit Dashboard

The API client can be used in your Streamlit pages:

```python
import streamlit as st
from src.utils.api_client import call_api_inference

# In your Streamlit page
if st.button("Score Transaction"):
    with st.spinner("Calling KNIME API..."):
        result = call_api_inference(
            endpoint_url=st.session_state.api_config['endpoint_url'],
            transaction_data=transaction_data,
            auth_type=st.session_state.api_config['auth_type'],
            api_key=st.session_state.api_config['api_key']
        )

        if result['success']:
            st.success(f"Fraud Score: {result['data']['fraud_score']}")
            st.metric("Latency", f"{result['latency_ms']}ms")
        else:
            st.error(f"API Error: {result['error']}")
```

## Support

For issues or questions:
- Check logs for detailed error messages
- Verify API endpoint is reachable
- Test with `curl` or Postman first
- Ensure authentication credentials are correct
- Check firewall/network settings

## API Client Reference

### `call_api_inference()`

Main function for calling external model APIs.

**Parameters:**
- `endpoint_url` (str): Full URL of API endpoint
- `transaction_data` (dict): Transaction features
- `auth_type` (str): Authentication type (default: "None")
- `api_key` (str): API key or token (default: "")
- `timeout` (int): Request timeout in seconds (default: 30)
- `content_type` (str): Content-Type header (default: "application/json")
- `custom_headers` (dict): Additional headers (default: None)

**Returns:**
- dict: Response with success status, data, and metadata

### `test_api_connection()`

Test API connection with a ping request.

**Parameters:**
Same as `call_api_inference()` except `transaction_data`

**Returns:**
- dict: Connection test results

### `build_headers()`

Build HTTP headers based on authentication type.

**Parameters:**
- `auth_type` (str): Authentication type
- `api_key` (str): API key or token
- `content_type` (str): Content-Type header (default: "application/json")
- `custom_headers` (dict): Additional headers (default: None)

**Returns:**
- dict: HTTP headers dictionary
