# Multi-stage build: frontend (Node.js) + backend (Python 3.12)
# Stage 1: build the React application
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python server serving both API and static frontend
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Install backend dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code and built frontend assets
COPY backend/ backend/
COPY --from=web /web/dist frontend/dist

# Start uvicorn on the port assigned by Render (default 8000)
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
