# Records API

The Records API allows you to create, read, update, and delete health records for migraine tracking.

## Endpoints

### List Records

Retrieve all health records for the authenticated user, optionally filtered by time range.

```
GET /records
```

**Query Parameters:**
- `start` (optional) - Start time in ISO8601 format (default: `1970-01-01T00:00:00Z`)
- `stop` (optional) - End time in ISO8601 format (default: `2100-01-01T00:00:00Z`)

**Example:**
```
GET /records?start=2025-01-01T00:00:00Z&stop=2025-01-31T23:59:59Z
```

**Response:** `200 OK`
```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "migraine_level": 4.5,
    "stress_level": 2.0,
    "had_trigger_food": true,
    "had_medication": false,
    "physical_activity_duration": 30,
    "sleep_duration": 420,
    "had_premonition": true,
    "record_time": "2025-01-15T14:30:00Z"
  }
]
```

**Response:** `204 No Content` (if no records found)

---

### Create Record

Create a new health record.

```
POST /records
```

**Request Body:**
```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "migraine_level": 4.5,
  "stress_level": 2.0,
  "had_trigger_food": true,
  "had_medication": false,
  "physical_activity_duration": 30,
  "sleep_duration": 420,
  "had_premonition": true,
  "record_time": "2025-01-15T14:30:00Z"
}
```

**Field Descriptions:**
- `id` (optional) - UUID for the record. Auto-generated if not provided.
- `migraine_level` (float) - Migraine intensity from 0.0 to 5.0 (default: 0.0)
- `stress_level` (float) - Stress level from 0.0 to 5.0 (default: 0.0)
- `had_trigger_food` (boolean) - Whether trigger foods were consumed (default: false)
- `had_medication` (boolean) - Whether medication was taken (default: false)
- `physical_activity_duration` (integer) - Duration in minutes (default: 0)
- `sleep_duration` (integer) - Duration in minutes (default: 0)
- `had_premonition` (boolean) - Whether premonition symptoms occurred (default: false)
- `record_time` (optional) - ISO8601 timestamp. Defaults to current UTC time if not provided.

**Response:** `201 Created`
```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Error Response:** `400 Bad Request`
```json
"Invalid uuid format"
```

---

### Get Record

Retrieve a specific health record by ID.

```
GET /records/{id}
```

**Path Parameters:**
- `id` - The UUID of the record

**Response:** `200 OK`
```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "migraine_level": 4.5,
  "stress_level": 2.0,
  "had_trigger_food": true,
  "had_medication": false,
  "physical_activity_duration": 30,
  "sleep_duration": 420,
  "had_premonition": true,
  "record_time": "2025-01-15T14:30:00Z"
}
```

**Error Response:** `404 Not Found`
```json
"404 Not Found"
```

---

### Update Record

Update an existing health record. You can update all fields including the record_time.

```
PUT /records/{id}
```

**Path Parameters:**
- `id` - The UUID of the record to update

**Request Body:**
```json
{
  "migraine_level": 3.0,
  "stress_level": 2.0,
  "had_trigger_food": true,
  "had_medication": true,
  "physical_activity_duration": 45,
  "sleep_duration": 480,
  "had_premonition": false,
  "record_time": "2025-01-15T14:30:00Z"
}
```

**Note:** 
- The `id` in the path cannot be changed
- If you change `record_time`, the old record is deleted and a new one is created (due to InfluxDB time-series nature)
- All fields should be provided (partial updates not supported)

**Response:** `200 OK`
```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Error Responses:**
- `404 Not Found` - Record doesn't exist
- `422 Unprocessable Entity` - Time mismatch and delete failed
- `400 Bad Request` - Invalid data format

---

### Delete Record

Delete a health record.

```
DELETE /records/{id}
```

**Path Parameters:**
- `id` - The UUID of the record to delete

**Response:** `204 No Content` (successful deletion, empty body)

**Error Response:** `404 Not Found`

---

## Data Model

### Record Object

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `id` | string (UUID) | Unique identifier | Auto-generated |
| `migraine_level` | float | Migraine intensity (0.0-10.0) | 0.0 |
| `stress_level` | float | Stress level (0.0-10.0) | 0.0 |
| `had_trigger_food` | boolean | Consumed trigger foods | false |
| `had_medication` | boolean | Took medication | false |
| `physical_activity_duration` | integer | Exercise duration (minutes) | 0 |
| `sleep_duration` | integer | Sleep duration (minutes) | 0 |
| `had_premonition` | boolean | Experienced premonition | false |
| `record_time` | string (ISO8601) | Timestamp of record | Current UTC time |

## Time Format

All timestamps must be in ISO8601 format with timezone:
- `2025-01-15T14:30:00Z` (UTC)
- `2025-01-15T14:30:00+01:00` (with timezone offset)

If no timezone is provided, UTC is assumed.

## Authorization

All endpoints require authentication. The user can only access their own records (enforced by the `userid` claim from the JWT token).
