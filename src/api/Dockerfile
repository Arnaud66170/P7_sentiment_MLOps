
# 1 - Base image
FROM python:3.10-slim

# 2 - Set work directory
WORKDIR /app

# 3 - Copy requirements & install
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 4 - Copy rest of the app
COPY . .

# 5 - Expose port
EXPOSE 8000

# 6 - Launch command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
