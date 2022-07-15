# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

EXPOSE 8004

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

ENV ACCESS_LOG=${ACCESS_LOG:-/proc/1/fd/1}

ENV ERROR_LOG=${ERROR_LOG:-/proc/1/fd/2}

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--worker-tmp-dir", "/dev/shm","--bind", "0.0.0.0:8004","-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--access-logfile", "$ACCESS_LOG", "--error-logfile", "$ERROR_LOG", "--log-file", "-"]
