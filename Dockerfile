FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY data/mock /app/data/mock

RUN pip install --upgrade pip && pip install . \
    && adduser --disabled-password --gecos "" clio

USER clio

ENTRYPOINT ["clio"]
CMD ["--help"]
