# Code Examples

This document provides practical examples for integrating with the Relief aHead API in different programming languages and environments.

## Configuration Constants

```javascript
// API Endpoints
const BASE_URL_PROD = 'https://api.reliefahead.com/v2';
const BASE_URL_TEST = 'https://api.reliefahead.com/test';

// Authentication
const AUTH_URL = 'https://cognito-idp.eu-north-1.amazonaws.com/';
const AUTH_TARGET = 'AWSCognitoIdentityProviderService.InitiateAuth';
const CLIENT_ID = '7do3cnirq5f10h1h6rv49c8n2s';
```

---

## Authentication

Before making API requests, authenticate to get an access token. See [AUTHENTICATION.md](AUTHENTICATION.md) for full details.

### cURL

```bash
curl -X POST https://cognito-idp.eu-north-1.amazonaws.com/ \
  -H "Content-Type: application/x-amz-json-1.1" \
  -H "X-Amz-Target: AWSCognitoIdentityProviderService.InitiateAuth" \
  -d '{
    "AuthFlow": "USER_PASSWORD_AUTH",
    "ClientId": "7do3cnirq5f10h1h6rv49c8n2s",
    "AuthParameters": {
      "USERNAME": "your-email@example.com",
      "PASSWORD": "your-password"
    }
  }'
```

Response:
```json
{
  "AuthenticationResult": {
    "AccessToken": "eyJraWQiOiJ...",
    "IdToken": "eyJraWQiOiJ...",
    "RefreshToken": "eyJjdHkiOiJ...",
    "ExpiresIn": 300,
    "TokenType": "Bearer"
  }
}
```

### Python

```python
import requests

# Configuration
AUTH_URL = "https://cognito-idp.eu-north-1.amazonaws.com/"
AUTH_TARGET = "AWSCognitoIdentityProviderService.InitiateAuth"
CLIENT_ID = "7do3cnirq5f10h1h6rv49c8n2s"

def login(username, password):
    """Login and get access token"""
    response = requests.post(AUTH_URL, 
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": AUTH_TARGET
        },
        json={
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": CLIENT_ID,
            "AuthParameters": {
                "USERNAME": username,
                "PASSWORD": password
            }
        }
    )
    
    if response.status_code == 200:
        return response.json()["AuthenticationResult"]["AccessToken"]
    raise Exception(f"Login failed: {response.json()}")

# Usage
access_token = login("your-email@example.com", "your-password")
```

### JavaScript

```javascript
// Configuration
const AUTH_URL = "https://cognito-idp.eu-north-1.amazonaws.com/";
const AUTH_TARGET = "AWSCognitoIdentityProviderService.InitiateAuth";
const CLIENT_ID = "7do3cnirq5f10h1h6rv49c8n2s";

async function login(username, password) {
  const response = await fetch(AUTH_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-amz-json-1.1",
      "X-Amz-Target": AUTH_TARGET
    },
    body: JSON.stringify({
      AuthFlow: "USER_PASSWORD_AUTH",
      ClientId: CLIENT_ID,
      AuthParameters: { USERNAME: username, PASSWORD: password }
    })
  });
  
  const result = await response.json();
  if (!response.ok) throw new Error(`Login failed: ${JSON.stringify(result)}`);
  
  return result.AuthenticationResult.AccessToken;
}

// Usage
const accessToken = await login("your-email@example.com", "your-password");

// Confirm Sign Up
async function confirmSignUp(email, code) {
  try {
    const response = await fetch(`${AUTH_CONFIG.authDomain}/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: email,
        confirmationCode: code
      })
    });
    console.log('Email confirmed');
    return await response.json();
  } catch (error) {
    console.error('Error confirming sign up:', error);
    throw error;
  }
}

// Sign In (OAuth 2.0 Resource Owner Password Flow)
async function signIn(email, password) {
  try {
    const response = await fetch(AUTH_CONFIG.tokenEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: new URLSearchParams({
        grant_type: 'password',
        username: email,
        password: password,
        client_id: '<your-client-id>'
      })
    });
    const tokens = await response.json();
    // tokens.access_token, tokens.refresh_token
    console.log('User signed in');
    return tokens;
  } catch (error) {
    console.error('Error signing in:', error);
    throw error;
  }
}

// Get JWT Token (from stored tokens)
function getToken() {
  // Retrieve from secure storage
  const tokens = JSON.parse(localStorage.getItem('auth_tokens'));
  return tokens?.access_token;
}

// Refresh Token
async function refreshToken(refreshToken) {
  try {
    const response = await fetch(AUTH_CONFIG.tokenEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: '<your-client-id>'
      })
    });
    const tokens = await response.json();
    return tokens;
  } catch (error) {
    console.error('Error refreshing token:', error);
    throw error;
  }
}

// Sign Out
async function signOut() {
  // Clear stored tokens
  localStorage.removeItem('auth_tokens');
  console.log('User signed out');
}
```

### Python (Complete Workflow)

```python
import requests

# Configuration
AUTH_URL = "https://cognito-idp.eu-north-1.amazonaws.com/"
AUTH_TARGET = "AWSCognitoIdentityProviderService.InitiateAuth"
CLIENT_ID = "7do3cnirq5f10h1h6rv49c8n2s"

def login(username, password):
    """Login and get access token"""
    response = requests.post(AUTH_URL, 
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": AUTH_TARGET
        },
        json={
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": CLIENT_ID,
            "AuthParameters": {
                "USERNAME": username,
                "PASSWORD": password
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()["AuthenticationResult"]
        return {
            "access_token": result["AccessToken"],
            "refresh_token": result["RefreshToken"],
            "expires_in": result["ExpiresIn"]
        }
    raise Exception(f"Login failed: {response.json()}")

def refresh_token(refresh_token):
    """Refresh access token"""
    response = requests.post(AUTH_URL,
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": AUTH_TARGET
        },
        json={
            "AuthFlow": "REFRESH_TOKEN_AUTH",
            "ClientId": CLIENT_ID,
            "AuthParameters": {
                "REFRESH_TOKEN": refresh_token
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()["AuthenticationResult"]
        return {
            "access_token": result["AccessToken"],
            "expires_in": result["ExpiresIn"]
        }
    raise Exception(f"Token refresh failed: {response.json()}")

# Example Usage
tokens = login("your-email@example.com", "your-password")
access_token = tokens["access_token"]

# When token expires (after 5 minutes), refresh it
new_tokens = refresh_token(tokens["refresh_token"])
access_token = new_tokens["access_token"]
```

---

## API Request Examples

### JavaScript (Fetch)

```javascript
const BASE_URL = 'https://api.reliefahead.com/v2';

// Get token from Auth (OAuth example above)
const token = await getToken();

// Create Record
async function createRecord(recordData) {
  const response = await fetch(`${BASE_URL}/records`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(recordData)
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// List Records
async function listRecords(startDate, endDate) {
  const params = new URLSearchParams({
    start: startDate || '1970-01-01T00:00:00Z',
    stop: endDate || '2100-01-01T00:00:00Z'
  });
  
  const response = await fetch(`${BASE_URL}/records?${params}`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (response.status === 204) {
    return []; // No records
  }
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Get Single Record
async function getRecord(recordId) {
  const response = await fetch(`${BASE_URL}/records/${recordId}`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Update Record
async function updateRecord(recordId, recordData) {
  const response = await fetch(`${BASE_URL}/records/${recordId}`, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(recordData)
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Delete Record
async function deleteRecord(recordId) {
  const response = await fetch(`${BASE_URL}/records/${recordId}`, {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
}

// Save Location
async function saveLocation(lat, lon) {
  const response = await fetch(`${BASE_URL}/location`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ lat, lon })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Get User Profile (use OAuth UserInfo endpoint instead)
// The /users/me endpoint is deprecated
// Use OAuth 2.0 UserInfo endpoint:
// const userInfo = await fetch('https://auth.reliefahead.com/oauth2/userInfo', {
//   headers: { 'Authorization': `Bearer ${token}` }
// }).then(r => r.json());

// Get Weather Data
async function getWeather(lat, lon) {
  const params = new URLSearchParams({ lat, lon });
  
  const response = await fetch(`${BASE_URL}/weather?${params}`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Save Weather Data
async function saveWeather(lat, lon, timestamp) {
  const response = await fetch(`${BASE_URL}/weather`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ lat, lon, timestamp })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Example Usage
async function exampleUsage() {
  // Create a record
  const newRecord = await createRecord({
    migraine_level: 7.5,
    stress_level: 6.0,
    had_trigger_food: true,
    had_medication: false,
    physical_activity_duration: 30,
    sleep_duration: 420,
    had_premonition: true,
    record_time: new Date().toISOString()
  });
  
  console.log('Created record:', newRecord);
  
  // List records from last 30 days
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  
  const records = await listRecords(
    thirtyDaysAgo.toISOString(),
    new Date().toISOString()
  );
  
  console.log('Records:', records);
  
  // Save current location
  await saveLocation(59.3293, 18.0686);
  console.log('Location saved');
}
```

### Python (requests)

```python
import requests
from datetime import datetime, timedelta

BASE_URL = 'https://api.reliefahead.com/v2'

# Assuming you have the access_token from authentication
access_token = 'your-jwt-token-here'

headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

# Create Record
def create_record(record_data):
    response = requests.post(
        f'{BASE_URL}/records',
        headers=headers,
        json=record_data
    )
    response.raise_for_status()
    return response.json()

# List Records
def list_records(start_date=None, stop_date=None):
    params = {
        'start': start_date or '1970-01-01T00:00:00Z',
        'stop': stop_date or '2100-01-01T00:00:00Z'
    }
    response = requests.get(
        f'{BASE_URL}/records',
        headers=headers,
        params=params
    )
    
    if response.status_code == 204:
        return []
    
    response.raise_for_status()
    return response.json()

# Get Single Record
def get_record(record_id):
    response = requests.get(
        f'{BASE_URL}/records/{record_id}',
        headers=headers
    )
    response.raise_for_status()
    return response.json()

# Update Record
def update_record(record_id, record_data):
    response = requests.put(
        f'{BASE_URL}/records/{record_id}',
        headers=headers,
        json=record_data
    )
    response.raise_for_status()
    return response.json()

# Delete Record
def delete_record(record_id):
    response = requests.delete(
        f'{BASE_URL}/records/{record_id}',
        headers=headers
    )
    response.raise_for_status()

# Save Location
def save_location(lat, lon):
    response = requests.post(
        f'{BASE_URL}/location',
        headers=headers,
        json={'lat': lat, 'lon': lon}
    )
    response.raise_for_status()
    return response.json()

# Get User Profile (use OAuth UserInfo endpoint instead)
# The /users/me endpoint is deprecated
# Use OAuth 2.0 UserInfo endpoint:
# response = requests.get(
#     'https://auth.reliefahead.com/oauth2/userInfo',
#     headers={'Authorization': f'Bearer {access_token}'}
# )
# user_info = response.json()

# Get Weather Data
def get_weather(lat, lon):
    params = {'lat': lat, 'lon': lon}
    response = requests.get(
        f'{BASE_URL}/weather',
        headers=headers,
        params=params
    )
    response.raise_for_status()
    return response.json()

# Save Weather Data
def save_weather(lat, lon, timestamp=None):
    data = {'lat': lat, 'lon': lon}
    if timestamp:
        data['timestamp'] = timestamp
    response = requests.post(
        f'{BASE_URL}/weather',
        headers=headers,
        json=data
    )
    response.raise_for_status()
    return response.json()


# Example Usage
if __name__ == '__main__':
    # Create a record
    new_record = create_record({
        'migraine_level': 7.5,
        'stress_level': 6.0,
        'had_trigger_food': True,
        'had_medication': False,
        'physical_activity_duration': 30,
        'sleep_duration': 420,
        'had_premonition': True,
        'record_time': datetime.utcnow().isoformat() + 'Z'
    })
    print(f"Created record: {new_record}")
    
    # List records from last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    records = list_records(
        start_date=thirty_days_ago.isoformat() + 'Z',
        stop_date=datetime.utcnow().isoformat() + 'Z'
    )
    print(f"Found {len(records)} records")
    
    # Save location
    result = save_location(59.3293, 18.0686)
    print(f"Location saved: {result}")
    
    # Get weather for location
    weather = get_weather(59.3293, 18.0686)
    print(f"Current weather: {weather}")
    
```

### cURL

```bash
# Set your token
TOKEN="your-jwt-token-here"
BASE_URL="https://api.reliefahead.com/v2"

# Create Record
curl -X POST "$BASE_URL/records" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "migraine_level": 7.5,
    "stress_level": 6.0,
    "had_trigger_food": true,
    "had_medication": false,
    "physical_activity_duration": 30,
    "sleep_duration": 420,
    "had_premonition": true,
    "record_time": "2025-01-15T14:30:00Z"
  }'

# List Records (with date filter)
curl -X GET "$BASE_URL/records?start=2025-01-01T00:00:00Z&stop=2025-01-31T23:59:59Z" \
  -H "Authorization: Bearer $TOKEN"

# Get Single Record
curl -X GET "$BASE_URL/records/a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
  -H "Authorization: Bearer $TOKEN"

# Update Record
curl -X PUT "$BASE_URL/records/a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "migraine_level": 8.0,
    "stress_level": 7.0,
    "had_trigger_food": true,
    "had_medication": true,
    "physical_activity_duration": 45,
    "sleep_duration": 480,
    "had_premonition": false,
    "record_time": "2025-01-15T14:30:00Z"
  }'

# Delete Record
curl -X DELETE "$BASE_URL/records/a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
  -H "Authorization: Bearer $TOKEN"

# Save Location
curl -X POST "$BASE_URL/location" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 59.3293,
    "lon": 18.0686
  }'

# Get User Profile (DEPRECATED - use OAuth UserInfo endpoint instead)
# curl -X GET "https://auth.reliefahead.com/oauth2/userInfo" \
#   -H "Authorization: Bearer $TOKEN"

# Get Weather Data
curl -X GET "$BASE_URL/weather?lat=59.3293&lon=18.0686" \
  -H "Authorization: Bearer $TOKEN"

# Save Weather Data
curl -X POST "$BASE_URL/weather" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 59.3293,
    "lon": 18.0686,
    "timestamp": "2025-01-15T14:30:00Z"
  }'
```

---

## Error Handling Example

```javascript
async function safeApiCall(apiFunction, ...args) {
  try {
    return await apiFunction(...args);
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.status);
      console.error('Error data:', error.response.data);
      
      switch (error.response.status) {
        case 400:
          console.error('Bad request - check your data format');
          break;
        case 401:
          console.error('Unauthorized - token expired or invalid');
          // Trigger re-authentication
          break;
        case 404:
          console.error('Resource not found');
          break;
        case 429:
          console.error('Rate limit exceeded - slow down requests');
          break;
        case 500:
          console.error('Server error - try again later');
          break;
      }
    } else if (error.request) {
      // Request made but no response
      console.error('No response from server - check network');
    } else {
      // Something else happened
      console.error('Error:', error.message);
    }
    throw error;
  }
}

// Usage
await safeApiCall(createRecord, recordData);
```
