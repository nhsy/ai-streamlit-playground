# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
# Check if running in a container
ENV DOCKER_CONTAINER=true

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
