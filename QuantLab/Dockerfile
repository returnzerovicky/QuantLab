# Use official slim Python image for performance and tiny footprint
FROM python:3.11-slim

# Set environment variables for clean operation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

# Set the working directory inside container
WORKDIR /app

# Install system dependencies needed for compiling numerical modules if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements file first to utilize Docker layer caching
COPY requirements.txt .

# Install dependencies with optimization flags
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source code
COPY . .

# Expose port (Streamlit default)
EXPOSE 8501

# Add standard healthcheck to ensure service availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
