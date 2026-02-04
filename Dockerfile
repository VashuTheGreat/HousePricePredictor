# Use the official Python image
FROM python:3.11-slim-buster

# Install uv inside the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy all project files
COPY . /app

# Synchronize dependencies using uv
RUN uv pip install --system -r requirements.txt

EXPOSE 8000

# Use python to run the main script (dependencies are already installed in system)
CMD ["python", "main.py"]
