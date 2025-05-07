# ML Image Classifier

A machine learning-based image classification application built with Python, TensorFlow, and Streamlit.

## Prerequisites

- Docker installed on your system
- Git (optional, for cloning the repository)

## Quick Start

### Running Locally with Docker

1. Clone the repository (or download the source code):
   ```bash
   git clone <repository-url>
   cd ml-ImageClassifier
   ```

2. Build the Docker image:
   ```bash
   docker build -t ml-imageclassifier .
   ```

3. Run the container:
   ```bash
   docker run -p 5000:5000 ml-imageclassifier
   ```

4. Access the application:
   Open your web browser and navigate to `http://localhost:5000`

### Docker Commands Reference

- Build the image:
  ```bash
  docker build -t ml-imageclassifier .
  ```

- Run the container:
  ```bash
  docker run -p 8501:8501 ml-imageclassifier
  ```

- Run in detached mode (background):
  ```bash
  docker run -d -p 8501:8501 ml-imageclassifier
  ```

- Stop the container:
  ```bash
  docker stop $(docker ps -q --filter ancestor=ml-imageclassifier)
  ```

- View running containers:
  ```bash
  docker ps
  ```

- View container logs:
  ```bash
  docker logs $(docker ps -q --filter ancestor=ml-imageclassifier)
  ```

## Deployment

### Publishing to Docker Hub

1. Create a Docker Hub account at https://hub.docker.com if you don't have one

2. Login to Docker Hub from your terminal:
   ```bash
   docker login
   ```

3. Tag your image with your Docker Hub username:
   ```bash
   docker tag ml-imageclassifier yourusername/ml-imageclassifier:latest
   ```

4. Push the image to Docker Hub:
   ```bash
   docker push yourusername/ml-imageclassifier:latest
   ```

5. Your image is now public! Others can pull and run it using:
   ```bash
   docker pull yourusername/ml-imageclassifier:latest
   docker run -p 8501:8501 yourusername/ml-imageclassifier:latest
   ```

### Deploying to a Cloud Platform

#### AWS EC2
1. Launch an EC2 instance with Docker installed
2. Copy your application files to the instance
3. Build and run the Docker container:
   ```bash
   docker build -t ml-imageclassifier .
   docker run -d -p 8501:8501 ml-imageclassifier
   ```

#### Google Cloud Run
1. Install and initialize Google Cloud SDK
2. Build and push the container:
   ```bash
   docker build -t gcr.io/[PROJECT-ID]/ml-imageclassifier .
   docker push gcr.io/[PROJECT-ID]/ml-imageclassifier
   ```
3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy ml-imageclassifier \
     --image gcr.io/[PROJECT-ID]/ml-imageclassifier \
     --platform managed \
     --port 8501
   ```

## Environment Variables

The application uses the following environment variables:
- `PYTHONUNBUFFERED=1` (set in Dockerfile)

## Troubleshooting

1. If port 8501 is already in use:
   ```bash
   # Find the process using the port
   netstat -ano | findstr :8501
   # Kill the process or use a different port
   ```

2. If the container fails to start:
   ```bash
   # Check container logs
   docker logs $(docker ps -q --filter ancestor=ml-imageclassifier)
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here] 