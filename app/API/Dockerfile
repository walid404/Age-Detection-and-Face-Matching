# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /Age-Detection-and-Face-Matching

# Copy requirements and install
COPY app/API/requirements.txt .
COPY ./src ./src
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the API folder and shared src
COPY app/API ./app/API
COPY src ./src

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app.API.main:app", "--host", "0.0.0.0", "--port", "8000"]