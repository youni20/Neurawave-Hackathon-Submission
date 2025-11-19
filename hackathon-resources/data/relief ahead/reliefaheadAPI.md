# Relief aHead Records API

This is the Relief aHead API for creating and managing migraine records. 

**For Hackathon Participants:** We have created specific accounts for the hackathon. If you need access to the app data, please ask us for login credentials.

API for managing health data in the Relief aHead system.

## API Documentation

The API is documented with the OpenAPI 3.0 specification in `openapi.yaml`.

### View API Documentation with Swagger UI

To start Swagger UI and view the API documentation:

```bash

# Then run Swagger UI
docker run -d -p 8080:8080 \
  -e SWAGGER_JSON=/openapi.yaml \
  -v "$(pwd)/openapi.yaml:/openapi.yaml" \
  swaggerapi/swagger-ui
```

Then open your browser at: **http://localhost:8080**

### Stop Swagger UI

To stop the Swagger UI container:

```bash
# List containers
docker ps

# Stop the container (use CONTAINER ID from the list above)
docker stop <CONTAINER_ID>

# Remove the container
docker rm <CONTAINER_ID>
```

Or stop all Swagger UI containers:

```bash
docker stop $(docker ps -q --filter ancestor=swaggerapi/swagger-ui)
docker rm $(docker ps -aq --filter ancestor=swaggerapi/swagger-ui)
```
