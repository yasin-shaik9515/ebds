FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libxkbcommon0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Test that app can import without errors
RUN python -c "import app; print('✓ App imports successfully')" 2>&1

# Copy application
COPY . .

# Download model during build
RUN python download_model.py

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
