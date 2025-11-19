# Authentication

Relief aHead uses AWS Cognito for user authentication and authorization. All API endpoints require a valid JWT token in the Authorization header.

## Authentication Endpoint

- **Cognito API:** `https://cognito-idp.eu-north-1.amazonaws.com/`
- **Client ID:** `7do3cnirq5f10h1h6rv49c8n2s`
- **Region:** `eu-north-1`

## How It Works

1. **Login** - Send username and password to Cognito API
2. **Get Tokens** - Receive AccessToken, IdToken, and RefreshToken
3. **Use Token** - Include AccessToken in API requests
4. **Refresh** - Use RefreshToken to get new tokens when expired (after 5 minutes)

**Include the AccessToken in all API requests:**

```http
Authorization: Bearer <AccessToken>
```

## Quick Start Examples

### 1. Login (Get Tokens)

#### cURL
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

#### Python
```python
import requests
import json

def login(username, password):
    url = "https://cognito-idp.eu-north-1.amazonaws.com/"
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"
    }
    payload = {
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7do3cnirq5f10h1h6rv49c8n2s",
        "AuthParameters": {
            "USERNAME": username,
            "PASSWORD": password
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    
    if response.status_code == 200:
        auth_result = result["AuthenticationResult"]
        return {
            "access_token": auth_result["AccessToken"],
            "id_token": auth_result["IdToken"],
            "refresh_token": auth_result["RefreshToken"],
            "expires_in": auth_result["ExpiresIn"]  # 300 seconds (5 minutes)
        }
    else:
        raise Exception(f"Login failed: {result}")

# Usage
tokens = login("your-email@example.com", "your-password")
print(f"Access Token: {tokens['access_token'][:50]}...")
```

#### JavaScript
```javascript
async function login(username, password) {
  const url = "https://cognito-idp.eu-north-1.amazonaws.com/";
  const headers = {
    "Content-Type": "application/x-amz-json-1.1",
    "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"
  };
  const payload = {
    AuthFlow: "USER_PASSWORD_AUTH",
    ClientId: "7do3cnirq5f10h1h6rv49c8n2s",
    AuthParameters: {
      USERNAME: username,
      PASSWORD: password
    }
  };
  
  const response = await fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify(payload)
  });
  
  const result = await response.json();
  
  if (response.ok) {
    const authResult = result.AuthenticationResult;
    return {
      accessToken: authResult.AccessToken,
      idToken: authResult.IdToken,
      refreshToken: authResult.RefreshToken,
      expiresIn: authResult.ExpiresIn  // 300 seconds (5 minutes)
    };
  } else {
    throw new Error(`Login failed: ${JSON.stringify(result)}`);
  }
}

// Usage
login("your-email@example.com", "your-password")
  .then(tokens => console.log("Access Token:", tokens.accessToken.substring(0, 50) + "..."))
  .catch(error => console.error(error));
```

### 2. Token Refresh

Access tokens expire after 5 minutes. Use the refresh token to get new tokens without re-entering credentials.

#### cURL
```bash
curl -X POST https://cognito-idp.eu-north-1.amazonaws.com/ \
  -H "Content-Type: application/x-amz-json-1.1" \
  -H "X-Amz-Target: AWSCognitoIdentityProviderService.InitiateAuth" \
  -d '{
    "AuthFlow": "REFRESH_TOKEN_AUTH",
    "ClientId": "7do3cnirq5f10h1h6rv49c8n2s",
    "AuthParameters": {
      "REFRESH_TOKEN": "your-refresh-token-here"
    }
  }'
```

#### Python
```python
import requests
import json

def refresh_tokens(refresh_token):
    url = "https://cognito-idp.eu-north-1.amazonaws.com/"
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"
    }
    payload = {
        "AuthFlow": "REFRESH_TOKEN_AUTH",
        "ClientId": "7do3cnirq5f10h1h6rv49c8n2s",
        "AuthParameters": {
            "REFRESH_TOKEN": refresh_token
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    
    if response.status_code == 200:
        auth_result = result["AuthenticationResult"]
        return {
            "access_token": auth_result["AccessToken"],
            "id_token": auth_result["IdToken"],
            "expires_in": auth_result["ExpiresIn"]  # 300 seconds
        }
    else:
        raise Exception(f"Token refresh failed: {result}")

# Usage
new_tokens = refresh_tokens(tokens["refresh_token"])
print(f"New Access Token: {new_tokens['access_token'][:50]}...")
```

#### JavaScript
```javascript
async function refreshTokens(refreshToken) {
  const url = "https://cognito-idp.eu-north-1.amazonaws.com/";
  const headers = {
    "Content-Type": "application/x-amz-json-1.1",
    "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"
  };
  const payload = {
    AuthFlow: "REFRESH_TOKEN_AUTH",
    ClientId: "7do3cnirq5f10h1h6rv49c8n2s",
    AuthParameters: {
      REFRESH_TOKEN: refreshToken
    }
  };
  
  const response = await fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify(payload)
  });
  
  const result = await response.json();
  
  if (response.ok) {
    const authResult = result.AuthenticationResult;
    return {
      accessToken: authResult.AccessToken,
      idToken: authResult.IdToken,
      expiresIn: authResult.ExpiresIn  // 300 seconds
    };
  } else {
    throw new Error(`Token refresh failed: ${JSON.stringify(result)}`);
  }
}

// Usage
refreshTokens(tokens.refreshToken)
  .then(newTokens => console.log("New Access Token:", newTokens.accessToken.substring(0, 50) + "..."))
  .catch(error => console.error(error));
```

## Token Structure

The IdToken is a JWT containing user claims:
- `sub` - Unique user ID (UUID)
- `email` - User email address
- `email_verified` - Email verification status
- Custom attributes (name, birthdate, etc.)

## Security Notes

- Never share tokens or store them insecurely
- Tokens should be stored securely on the client (e.g., secure storage, not localStorage for web)
- Implement token refresh before expiration (IdToken/AccessToken expire after just a few minutes)
- Use HTTPS for all requests
- Tokens are JWT format and should be validated

## Client Libraries

**For AWS Cognito Integration:**

**JavaScript/TypeScript:**
- `aws-amplify` - Full AWS Amplify SDK
- `amazon-cognito-identity-js` - Cognito-specific SDK
- Native `fetch` - Use examples in this documentation

**Python:**
- `requests` - Use examples in this documentation (recommended for simplicity)
- `boto3` - AWS SDK for Python

**Swift (iOS):**
- `AWSCognitoIdentityProvider` - AWS SDK for iOS
- Native URLSession - Use similar approach as JavaScript examples

**Kotlin (Android):**
- `AWS SDK for Android` - Official AWS SDK
- Native HTTP client - Use similar approach as JavaScript examples

**Note:** Standard OAuth 2.0 libraries may not work directly with the Cognito Identity Provider API. Use the examples in this documentation or AWS SDKs.
