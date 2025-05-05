FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Collect static and run migrations
RUN python manage.py migrate
RUN python manage.py collectstatic --noinput

# Run app with Gunicorn
CMD gunicorn agro_diagnosis.wsgi:application --bind 0.0.0.0:8080 --workers=3