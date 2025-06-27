# Raspberry Pi Bullseye 64bit 환경용 Python 3.8 슬림 베이스 이미지
FROM arm64v8/python:3.8-slim-bullseye

# 1) 비대화식 모드 설정 (APT 설치 시 대화창 방지)
ENV DEBIAN_FRONTEND=noninteractive

# 2) 필수 패키지 설치 및 저장소 등록
RUN set -eux; \
    # 기본 패키지 설치
    apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        build-essential; \
    \
    # Coral EdgeTPU 저장소 GPG 키 추가
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        | tee /etc/apt/sources.list.d/coral-edgetpu.list; \
    \
    # Raspberry Pi 저장소 GPG 키 추가
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.org/debian bullseye main" \
        | tee /etc/apt/sources.list.d/raspi.list; \
    \
    # 저장소 업데이트
    apt-get update; \
    \
    # libcamera 및 추가 의존성 패키지 설치
    apt-get install -y --no-install-recommends \
        libboost-dev libgnutls28-dev openssl libtiff-dev pybind11-dev qtbase5-dev libqt5core5a libqt5widgets5 meson cmake python3-yaml python3-ply \
        libglib2.0-dev libgstreamer-plugins-base1.0-dev git python3-pip \
        ninja-build pkg-config libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libdrm-dev libexif-dev libudev-dev libyaml-cpp-dev python3-jinja2 libboost-program-options-dev libpng-dev; \
    \
    # libcamera 빌드에 필요한 Python 모듈 설치
    pip install --no-cache-dir \
        jinja2 \
        pyyaml \
        ply; \
    \
    # APT 캐시 정리
    rm -rf /var/lib/apt/lists/*

# 3) libcamera 소스 클론 및 빌드
RUN git clone https://git.libcamera.org/libcamera/libcamera.git /libcamera && \
    sed -i '/removeprefix/s/\(.*\)= kebab_case(\(.*\)\.removeprefix(\(.*\)))$/\1 = kebab_case((\2[len(\3):] if \2.startswith(\3) else \2))/' /libcamera/utils/codegen/gen-gst-controls.py && \
    cd /libcamera && \
    meson setup build && \
    ninja -C build && \
    ninja -C build install && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/libcamera.conf && ldconfig

# 4) libcamera-apps 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends libcamera-apps

# 5) python 명령어를 python3와 동일하게 연결
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --set python /usr/bin/python3

# 6) pip 최신화 및 추가 파이썬 라이브러리 설치
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        tqdm \
        pillow \
        pycocotools \
        pyyaml \
        redis \
        tflite-runtime==2.11.0

# ── 애플리케이션 복사 & 기본 실행 ───────────────────────────────────────────
WORKDIR /app
COPY one_img_processor.py .
# COPY vision_processor.py .
ENTRYPOINT ["python", "-u", "one_img_processor.py"]
# ENTRYPOINT ["python", "-u", "vision_processor.py"]
