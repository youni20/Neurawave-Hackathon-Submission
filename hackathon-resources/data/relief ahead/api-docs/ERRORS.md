# Error Handling

This document describes error responses, status codes, and best practices for handling errors when using the Relief aHead API.

## HTTP Status Codes

### Success Codes

| Code | Name | Description |
|------|------|-------------|
| 200 | OK | Request succeeded, response body contains data |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request succeeded, no response body (e.g., successful delete or empty list) |

### Client Error Codes

| Code | Name | Description |
|------|------|-------------|
| 400 | Bad Request | Invalid request format or data |
| 401 | Unauthorized | Missing or invalid authentication token |
| 404 | Not Found | Requested resource doesn't exist |
| 422 | Unprocessable Entity | Valid request but cannot be processed (e.g., time mismatch on update) |
| 429 | Too Many Requests | Rate limit exceeded |

### Server Error Codes

| Code | Name | Description |
|------|------|-------------|
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Service temporarily unavailable |

---

## Error Response Format

Most error responses include a JSON body with error details:

```json
{
  "error": "Error message describing what went wrong"
}
```

Or plain text string:
```json
"Error message"
```

---

## Common Error Scenarios

### Authentication Errors

**401 Unauthorized**

Occurs when:
- No Authorization header provided
- Token is expired
- Token is invalid or malformed
- Token signature verification failed

**Response:**
```json
{
  "message": "Unauthorized"
}
```

**Solution:**
- Check that Authorization header is present
- Refresh the token if expired
- Re-authenticate the user

---

### Validation Errors

**400 Bad Request - Invalid UUID**

```json
"Invalid uuid format"
```

**Cause:** The `id` field contains an invalid UUID format.

**Solution:** Ensure UUIDs follow the standard format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

---

**400 Bad Request - Invalid Datetime**

```json
"Invalid datetime format 2025-13-45T99:99:99"
```

**Cause:** The `record_time` field contains an invalid ISO8601 timestamp.

**Solution:** Use valid ISO8601 format: `2025-01-15T14:30:00Z`

---

**400 Bad Request - Missing Required Fields**

For Location API:
```json
{
  "error": "latitude and longitude are required"
}
```

**Solution:** Include all required fields in the request body.

---

### Resource Errors

**404 Not Found**

```json
"404 Not Found"
```

**Cause:** 
- Record with the specified ID doesn't exist
- Record belongs to a different user
- Invalid endpoint URL

**Solution:**
- Verify the record ID is correct
- Check that the user has access to this resource
- Verify the endpoint URL

---

### Update Errors

**422 Unprocessable Entity - Time Mismatch**

```json
"Record time 2025-01-15T14:30:00Z in the body does not match the existing record time 2025-01-15T12:00:00Z. Delete failed: ..."
```

**Cause:** When updating a record with a different `record_time`, the system tries to delete the old record and create a new one (InfluxDB limitation). If the delete fails, this error occurs.

**Solution:** 
- Ensure the `record_time` matches the existing record if you don't want to change it
- If changing time, ensure the old record exists and is accessible

---

### Rate Limiting

**429 Too Many Requests**

**Limits:**
- Test environment: 1 request/second
- Production environment: 100 requests/second

**Response:** (HTTP 429 from API Gateway)
```json
{
  "message": "Too Many Requests"
}
```

**Solution:**
- Implement exponential backoff
- Reduce request frequency
- Batch operations when possible
- Cache responses where appropriate

---

### Server Errors

**500 Internal Server Error**

Generic error responses:
```json
"Error creating record: [error details]"
```
```json
"Error getting record: [error details]"
```
```json
"Error deleting record: [error details]"
```
```json
"500 [error details]"
```

**Cause:** Unexpected server-side error (database connection, internal processing, etc.)

**Solution:**
- Retry the request after a short delay
- If error persists, contact support
- Check service status

---

## Best Practices

### 1. Always Check Status Codes

```javascript
if (response.status >= 200 && response.status < 300) {
  // Success
} else if (response.status >= 400 && response.status < 500) {
  // Client error - fix the request
} else if (response.status >= 500) {
  // Server error - retry or report
}
```

### 2. Handle Token Expiration

```javascript
async function apiCallWithRetry(apiFunction, ...args) {
  try {
    return await apiFunction(...args);
  } catch (error) {
    if (error.response?.status === 401) {
      // Token expired, refresh it
      await refreshToken();
      // Retry the request
      return await apiFunction(...args);
    }
    throw error;
  }
}
```

### 3. Implement Exponential Backoff

```javascript
async function exponentialBackoff(apiFunction, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await apiFunction();
    } catch (error) {
      if (error.response?.status === 429 || error.response?.status >= 500) {
        const delay = Math.pow(2, i) * 1000; // 1s, 2s, 4s
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }
      throw error;
    }
  }
  throw new Error('Max retries exceeded');
}
```

### 4. Validate Data Before Sending

```javascript
function validateRecord(record) {
  if (record.id && !isValidUUID(record.id)) {
    throw new Error('Invalid UUID format');
  }
  
  if (record.migraine_level < 0 || record.migraine_level > 5) {
    throw new Error('migraine_level must be between 0 and 10');
  }
  
  if (record.record_time && !isValidISO8601(record.record_time)) {
    throw new Error('Invalid timestamp format');
  }
  
  // ... more validations
}
```

### 5. Handle Empty Responses

```javascript
const response = await fetch(`${BASE_URL}/records`);

if (response.status === 204) {
  // No records found
  return [];
}

const data = await response.json();
return data;
```

### 6. Log Errors for Debugging

```javascript
try {
  await createRecord(data);
} catch (error) {
  console.error('Failed to create record:', {
    status: error.response?.status,
    message: error.message,
    data: error.response?.data,
    timestamp: new Date().toISOString()
  });
  throw error;
}
```

---

## Error Monitoring

For production applications, consider:
- Implementing centralized error logging (e.g., Sentry, LogRocket)
- Tracking error rates and patterns
- Setting up alerts for high error rates
- Monitoring token expiration rates
- Tracking rate limit hits

---

## Support

If you encounter errors that are not documented here or persist despite following the guidelines:
- Check the API status page
- Review your implementation against the examples
- Contact support with error details and timestamps
