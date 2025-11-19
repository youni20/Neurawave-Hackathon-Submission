# Weather API

The Weather API is a proxy service to provides access to weather data that can be correlated with health records to identify potential environmental triggers for migraines.

## Endpoints

### Get Weather Data

Retrieve current weather data for a specific location.

```
GET /weather?lat={latitude}&lon={longitude}
```

**Query Parameters:**

- `lat` (required) - Latitude as a float (-90.0 to 90.0)
- `lon` (required) - Longitude as a float (-180.0 to 180.0)

**Example:**

```
GET /weather?lat=59.3293&lon=18.0686
```

**Response:** `200 OK`

```json
{
  "location": { "name", "country", "state" },
  "coordinates": { "lat", "lon" },
  "forecast": [ {...}, {...}, ... ],
  "city_info": { "id", "name", "coord", ... }
}
```

Full example: [api-weather-example.json](api-weather-example.json)

---

### Save Weather Data

Save weather data associated with a health record or location.

```
POST /weather
```

**Request Body:**

```json
{
  "lat": 59.3293,
  "lon": 18.0686,
  "timestamp": "2025-01-15T14:30:00Z"
}
```

**Field Descriptions:**

- `lat` (required) - Latitude as a float
- `lon` (required) - Longitude as a float
- `timestamp` (optional) - ISO8601 timestamp. Defaults to current UTC time if not provided.

The system will automatically fetch and store weather data for the specified location and time.

**Response:** `201 Created`

```json
{
  "id": "1a3e316a-08da-409e-9fd7-4dd94f953b5a"
}
```

**Error Response:** `400 Bad Request`

```json
{
  "error": "latitude and longitude are required"
}
```

---

## Response Data Fields

### Location Object

| Field | Type | Description |
|-------|------|-------------|
| `lat` | float | Latitude coordinate |
| `lon` | float | Longitude coordinate |
| `name` | string | Location name (city) |
| `country` | string | Country code (ISO 3166-1 alpha-2) |

### Current Weather Object

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `timestamp` | string | ISO8601 | Time of weather observation |
| `temperature` | float | °C | Current temperature |
| `feels_like` | float | °C | Perceived temperature |
| `pressure` | integer | hPa | Atmospheric pressure at sea level |
| `humidity` | integer | % | Relative humidity |
| `weather` | string | - | Weather condition (clear, cloudy, rain, etc.) |
| `weather_description` | string | - | Detailed weather description |
| `wind_speed` | float | m/s | Wind speed |
| `wind_direction` | integer | degrees | Wind direction (meteorological) |

## Weather Conditions

Common weather condition values:

- `clear` - Clear sky
- `clouds` - Cloudy
- `rain` - Rain
- `drizzle` - Light rain
- `thunderstorm` - Thunderstorm
- `snow` - Snow
- `mist` / `fog` - Reduced visibility
- `smoke` / `haze` / `dust` - Air quality issues

## Potential Migraine Triggers

Weather factors that may trigger migraines:

- **Pressure changes** - Rapid changes in atmospheric pressure
- **Temperature fluctuations** - Sudden temperature changes
- **High humidity** - Elevated moisture levels (>70%)
- **Storm systems** - Approaching thunderstorms
- **Bright conditions** - Clear, sunny weather with high UV
- **Wind** - Strong or gusty winds

## Data Source

Weather data is sourced from OpenWeatherMap API, providing:

- Current weather conditions
- Historical weather data
- Real-time updates
- Global coverage

## Authorization

All endpoints require authentication. Weather data is associated with the authenticated user for correlation with health records.

## Rate Limiting

Weather API calls are subject to the same rate limits as other endpoints:

- **Test environment:** 1 request/second
- **Production environment:** 100 requests/second

## Use Cases

- Correlate weather conditions with migraine occurrences
- Identify weather-related migraine triggers
- Track patterns across different locations
- Predict potential migraine episodes based on weather forecasts
- Monitor environmental factors during migraine events

## Privacy

Weather data is:

- Associated with the authenticated user
- Linked to locations where health records are created
- Used for personal health tracking only
- Not shared with other users
