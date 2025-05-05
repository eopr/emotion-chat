# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system packages needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the app (change if your entry point is different)
CMD ["python", "app.py"]
