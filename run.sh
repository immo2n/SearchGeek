#!/bin/bash
source embed-env/bin/activate
cd embed-service
echo "Starting FastAPI server..."
uvicorn app:app --host 0.0.0.0 --port 8001