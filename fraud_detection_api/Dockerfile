# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask API
EXPOSE 5000

# Run the Flask application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "serve_model:app"]
