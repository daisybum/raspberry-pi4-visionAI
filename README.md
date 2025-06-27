# Raspberry Pi 5 Vision AI

라즈베리파이 5를 위한 최적화된 비전 AI 시스템입니다. Edge TPU 없이 라즈베리파이 5의 CPU만으로 세그멘테이션과 분류 모델을 실행합니다.

## 시스템 구성

이 프로젝트는 다음과 같은 구성 요소로 이루어져 있습니다:

1. **Redis 서버**: 이미지 데이터 저장 및 서비스 간 통신을 위한 Pub/Sub 메시징 시스템
2. **추론 서버**: 세그멘테이션 및 분류 모델을 실행하는 서버 (Edge TPU 대신 라즈베리파이 5 CPU 사용)
3. **캡처 애플리케이션**: 카메라에서 이미지를 주기적으로 캡처하여 Redis에 게시
4. **추론 클라이언트**: 이미지 처리 요청을 보내고 결과를 처리하는 클라이언트

## 주요 특징

- **Edge TPU 의존성 제거**: 라즈베리파이 5의 CPU만으로 추론을 수행
- **최적화된 TFLite 모델**: 양자화 및 최적화를 통해 라즈베리파이 5에서 빠른 추론 가능
- **통합 추론 엔드포인트**: 세그멘테이션과 분류를 한 번의 요청으로 처리하는 `/combined` 엔드포인트 제공
- **멀티스레딩 최적화**: 라즈베리파이 5의 멀티코어 CPU를 활용한 병렬 처리

## 설치 및 실행

### 1. 모델 생성

먼저 최적화된 TFLite 모델을 생성합니다:

```bash
cd models
python create_optimized_models.py
```

이 스크립트는 다음 두 가지 모델을 생성합니다:
- `seg_model.tflite`: 세그멘테이션 모델
- `cls_model.tflite`: 분류 모델

### 2. Docker Compose 설정

새로운 Docker Compose 설정을 적용합니다:

```bash
mv docker-compose.yml.new docker-compose.yml
```

### 3. 시스템 실행

Docker Compose를 사용하여 전체 시스템을 실행합니다:

```bash
docker-compose up -d
```

## 구성 요소 상세 설명

### 추론 서버 (`inference_server/main.py`)

- TensorFlow Lite 런타임을 사용하여 최적화된 모델 실행
- 세 가지 엔드포인트 제공:
  - `/segment`: 이미지 세그멘테이션
  - `/classify`: 이미지 분류
  - `/combined`: 세그멘테이션과 분류를 동시에 수행
- 멀티스레딩을 통한 성능 최적화

### 추론 클라이언트 (`inference_client/run_client.py`)

- Redis Pub/Sub을 통해 새 이미지 알림을 수신
- 환경 변수 `INFERENCE_MODE`를 통해 추론 모드 선택 가능 (combined, segment, classify)
- 추론 시간 및 결과 상세 로깅

### 모델 최적화 (`models/create_optimized_models.py`)

- MobileNetV2 기반의 경량 모델 생성
- 양자화 및 최적화를 통한 추론 속도 향상
- 라즈베리파이 5 CPU에 최적화된 TFLite 모델 생성

## 성능 최적화 팁

1. **NUM_THREADS 조정**: 라즈베리파이 5의 코어 수에 맞게 `NUM_THREADS` 환경 변수 조정
2. **모델 크기 축소**: 필요에 따라 `create_optimized_models.py`에서 모델 입력 크기 조정
3. **캡처 주기 조정**: 필요에 따라 `INTERVAL` 환경 변수를 조정하여 캡처 주기 변경

## API 엔드포인트

### 세그멘테이션 (`/segment`)

```json
POST /segment
{
  "redis_key": "img:12345"
}
```

응답:
```json
{
  "mask_shape": [224, 224],
  "unique_labels": [0, 1],
  "inference_time_ms": 150
}
```

### 분류 (`/classify`)

```json
POST /classify
{
  "redis_key": "img:12345"
}
```

응답:
```json
{
  "id": 3,
  "score": 0.92,
  "inference_time_ms": 120
}
```

### 통합 추론 (`/combined`)

```json
POST /combined
{
  "redis_key": "img:12345"
}
```

응답:
```json
{
  "segmentation": {
    "mask_shape": [224, 224],
    "unique_labels": [0, 1]
  },
  "classification": {
    "id": 3,
    "score": 0.92
  },
  "inference_time_ms": 200
}
```
