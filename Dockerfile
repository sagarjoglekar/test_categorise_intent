# Use a Python base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

COPY download_data.py .

# Install project dependencies
RUN pip install -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_md

# Download spaCy models
RUN wget -O /app/spacy_classifier.joblib https://www.dropbox.com/s/a5x3epz6njqh42y/spacy_classifier.joblib?dl=0

RUN wget -O /app/TfIdfClassifier.joblib https://www.dropbox.com/s/0e109y5di4z326m/TfIdfClassifier.joblib?dl=0

# Copy the project files
COPY . .

# Expose a port (if needed)
EXPOSE 8000

# Run the model script as a service
CMD ["python", "src/model_service.py"]
