FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN \
    #apt --fix-broken -y install && \
    apt-get update && \
    apt remove python-pip  python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev\ 
    libtiff-dev \
    libwebp-dev \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3.8 1

# set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY ./src/ /app/

EXPOSE 8000
CMD ["celery", "-A", "worker", "worker", "--loglevel=info", "--pool=solo"]