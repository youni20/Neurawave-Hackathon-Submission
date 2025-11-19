# Location API

The Location API allows you to save GPS coordinates for the authenticated user. This can be used for tracking location-based triggers or patterns.

## Endpoints

### Save Location

Save the user's current GPS coordinates.

```
POST /location
```

**Request Body:**
```json
{
  "lat": 59.3293,
  "lon": 18.0686
}
```

**Field Descriptions:**
- `lat` (required) - Latitude as a float (-90.0 to 90.0)
- `lon` (required) - Longitude as a float (-180.0 to 180.0)

**Response:** `201 Created`
```json
{
  "message": "Location saved"
}
```

**Error Responses:**

`400 Bad Request` - Missing required fields
```json
{
  "error": "latitude and longitude are required"
}
```

`500 Internal Server Error` - Server error
```json
{
  "error": "Error message details"
}
```

---

## Data Storage

Locations are stored in a time-series database with:
- Timestamp (automatically set to current UTC time)
- User ID (from authentication token)
- Latitude
- Longitude

## Usage Notes

- Each location save creates a new timestamped entry
- Historical locations are preserved (no updates or deletes)
- Locations are tied to the authenticated user
- No endpoint to retrieve locations is currently exposed in the public API

## Privacy

Location data is:
- Associated only with the authenticated user
- Not shared with other users
- Stored securely in the backend database
- Can only be saved, not retrieved via the public API

## Example Use Cases

- Track location when migraine symptoms occur
- Correlate migraines with specific locations
- Identify environmental triggers based on location patterns

## Authorization

Requires authentication. The user can only save their own location data (enforced by the JWT token).
