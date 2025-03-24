# 1 - Base image
FROM python:3.10-slim

# 2 - Set work directory
WORKDIR /app

# 3 - Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 4 - Copy requirements & install
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5 - Copy rest of the app
COPY . .

# 6 - Expose port
EXPOSE 8000

# 7 - Launch command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

