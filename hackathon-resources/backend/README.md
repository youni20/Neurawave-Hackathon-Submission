# Backend Example

A simple Flask API example that you can use as a starting point for building your own backend.

## Getting Started

1. Navigate to the backend folder:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

3. Test the API:
```bash
curl http://localhost:8000/
```

The server will run on `http://localhost:8000`

## Extending the API

You can add more endpoints to serve the synthetic data, implement authentication, or connect to a database. Use the OpenAPI specification in `data/relief ahead/openapi.yaml` as inspiration for your API structure.
