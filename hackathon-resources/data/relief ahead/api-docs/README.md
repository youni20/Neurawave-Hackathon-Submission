# Relief aHead API Documentation

> **Note:** This is public API documentation intended for external developers. It describes only the publicly available endpoints and does not reflect the complete backend infrastructure.

Welcome to the Relief aHead API documentation. This REST API enables mobile and web applications to manage health data for migraine tracking and analysis.

## 🚀 Quick Start

**Base URLs:**
- Production: `https://api.reliefahead.com/v2`
- Test/Staging: `https://api.reliefahead.com/test`

**Getting Started:**
1. Read [Authentication](AUTHENTICATION.md) to set up OAuth 2.0 authentication
2. Explore [Code Examples](EXAMPLES.md) for implementation in JavaScript, Python, and cURL
3. Review API endpoints below and start building!

**All API requests require authentication.** Include your JWT token in the `Authorization` header:
```
Authorization: Bearer <your-jwt-token>
```

---

## 📚 API Endpoints

### [Authentication](AUTHENTICATION.md)
OAuth 2.0 / OpenID Connect authentication for user signup, login, and token management.

**Key endpoints:**
- Sign up new users
- Authenticate and receive JWT tokens
- Refresh expired tokens

---

### [Health Records](API-RECORDS.md)
Create, read, update, and delete migraine health records.

**Endpoints:**
- `POST /records` - Create a new health record
- `GET /records` - List all records (with time filtering)
- `GET /records/{id}` - Get a specific record
- `PUT /records/{id}` - Update a record
- `DELETE /records/{id}` - Delete a record

**Data tracked:**
- Migraine level (0-5)
- Stress level (0-5)
- Trigger food, medication, premonition (boolean)
- Physical activity duration (minutes)
- Sleep duration (minutes)

---

### [Location Tracking](API-LOCATION.md)
Save GPS coordinates to correlate location with health events.

**Endpoints:**
- `POST /location` - Save current GPS coordinates

**Use case:** Identify location-based migraine triggers.

---

### [Weather Data](API-WEATHER.md)
Access weather data to identify environmental migraine triggers.

**Endpoints:**
- `GET /weather` - Get current weather for a location
- `POST /weather` - Save weather data with timestamp

**Data includes:**
- Temperature, pressure, humidity
- Weather conditions
- Wind speed and direction

**Use case:** Correlate weather patterns with migraine occurrences.

---

## 📖 Additional Resources

- **[Code Examples](EXAMPLES.md)** - Ready-to-use code in JavaScript, Python, and cURL
- **[Error Handling](ERRORS.md)** - HTTP status codes, error responses, and best practices

## ⚡ Rate Limits

| Environment | Rate Limit |
|-------------|------------|
| Test/Staging | 1 request/second |
| Production | 100 requests/second |

**Tip:** Implement exponential backoff when you receive `429 Too Many Requests` errors.

## 🔐 Security

- All endpoints require HTTPS
- JWT tokens expire after 5 minutes (refresh tokens valid for 365 days)
- Store tokens securely (never in localStorage for web apps)
- Validate tokens on the client side

## 💬 Support

For questions or issues, contact the development team or file an issue in the appropriate repository.

---

**Last Updated:** 2025-11-19
