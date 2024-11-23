# Example Dockerfile for Python with PostgreSQL
FROM python:3.9-slim

# Install any necessary system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy project files
WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
 

EXPOSE 8005

CMD ["uvicorn", "chat:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]