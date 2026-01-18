# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /Age-Detection-and-Face-Matching

# Copy requirements and install
COPY app/FlaskUI/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy FlaskUI folder
COPY app/FlaskUI ./app/FlaskUI

# Expose Flask port
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=app.FlaskUI.main
ENV FLASK_ENV=development

# Run Flask
CMD ["python", "-m", "app.FlaskUI.main"]
