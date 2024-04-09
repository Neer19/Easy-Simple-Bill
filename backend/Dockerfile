# # Base image
# FROM ubuntu:20.04

# # Set working directory
# WORKDIR /app

# # Copy requirements.txt
# COPY requirements.txt .

# RUN apt-get update && apt-get -y install python3-pip python3-dev

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the app code
# COPY . .

# # Expose port (optional)
# EXPOSE 8000

# # Start the app
# CMD ["python", "app.py"]


FROM python:3.9.13

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install libgl1 -y

COPY . .

ENV FLASK_APP=app.py

# EXPOSE 8000

# CMD ["python","app.py"]

CMD flask run -h 0.0.0.0 -p 8000