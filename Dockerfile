# Use a Python base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install project dependencies
RUN pip install -r requirements.txt

# Copy the project files
COPY . .

# Expose a port (if needed)
EXPOSE 8000

# Run the model script as a service
CMD ["python", "src/model_service.py"]
