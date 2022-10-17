# SR-Worker

Serving Super Resolution Model Using FastAPI and Celery.

## For Developers

1. install dev package.

```shell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. install pre-commit.

```shell
pre-commit install
```

## Installation
1. Run RabbitMQ image as a broker
```shell
docker run -d --name sr-rabbitmq -p 5672:5672 -p 8080:15672 --restart=unless-stopped rabbitmq:3.9.21-management
```

2. Build docker image
```shell
git clone https://github.com/ainize-team/SR-Worker.git
cd SR-Worker
docker build -t sr-worker .
```

3. Run docker image
```shell
docker run -d --name <worker_container_name> \
--gpus='"device=0"' -e BROKER_URI=<broker_uri> \
-e DATABASE_URL=<firebase_realtime_database_url> \
-e STORAGE_BUCKET=<firebase_storage_url> \
-v <firebase_credential_path>:/app/key -v <model_local_path>:/app/model \
sr-worker
```
Or, you can use the env file to run as follows.
```shell
docker run -d --name <worker_container_name> \
--gpus='"device=0"' \
--env-file <env filename> \
-v <firebase_credential_path>:/app/key -v <model_local_path>:/app/model \
sr-worker
```

## Test with FastAPI
- Check our [SR-FastAPI](https://github.com/ainize-team/SR-FastAPI) Repo.
