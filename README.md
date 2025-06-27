# Raspberry Pi 5 Vision AI Pipeline

초경량 TFLite 세그멘테이션 + 분류 모델을 **라즈베리파이 5**의 CPU만으로 실행하는 Edge AI 프로젝트입니다. `vision_app` 컨테이너가 카메라 이미지를 주기적으로 캡처·추론하고, 결과를 **Redis** 에 게시하여 다른 마이크로서비스가 쉽게 구독할 수 있도록 설계되었습니다.

<p align="center">
  <img src="https://raw.githubusercontent.com/daisybum/raspberry-pi4-visionAI/main/.docs/arch.svg" alt="architecture" width="650" />
</p>

---

## 주요 특징

* **Edge TPU 불필요** – 순수 CPU(TFLite XNNPACK delegate) 만으로 2-3 FPS 달성
* **이중 추론** – 세그멘테이션 + 분류 모델을 한 번에 호출하여 네트워크 오버헤드 최소화
* **경량 컨테이너** – Slim Python + `tflite-runtime` 로 이미지 크기 &lt; 120 MB
* **Redis Pub/Sub** – 결과를 `result:<uuid>` 키로 TTL 1h 저장, 다양한 언어 클라이언트와 호환
* **유연한 모드**
  * `vision_processor.py` – 실기 버전(카메라 필요, 캡처 주기 ENV로 조정)
  * `one_img_processor.py` – 개발/디버깅 버전(샘플 이미지 반복 추론)
* **간단한 모델 교체** – `models/` 폴더에 새 TFLite 업로드 후 ENV 경로만 변경

---

## 빠른 시작

### 1. 요구 사항

* Raspberry Pi 4 (Bullseye 64-bit)
* Docker & Docker Compose
* CSI 카메라 모듈 (실기 모드)

```bash
# 설치 예시 (Raspberry Pi)
sudo apt update && sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER && newgrp docker
```

### 2. 프로젝트 클론

```bash
git clone https://github.com/daisybum/raspberry-pi4-visionAI.git
cd raspberry-pi4-visionAI
```

### 3. 컨테이너 빌드 & 실행

```bash
docker-compose up --build -d
```

> 📌 **TIP**: 라즈베리파이 없이 노트북에서 테스트하려면 `docker-compose.yml` 대신 `one_img_processor.py` 를 직접 실행해 보세요.

```bash
python -m pip install -r vision_app/requirements.txt
python vision_app/one_img_processor.py
```

---

## 디렉터리 구조

```
│  docker-compose.yml     # Redis + Vision App 스택 정의
│  README.md              # (현재 파일)
│
├─data/
│      example.jpg        # 샘플 이미지
│
├─models/
│      seg_model_int8.tflite
│      cls_model_int8.tflite
│
└─vision_app/
        Dockerfile        # Slim Python 기반 실행 이미지
        requirements.txt  # 런타임 의존성
        vision_processor.py   # 카메라 캡처 + 추론 (실기)
        one_img_processor.py  # 이미지 파일 반복 추론 (개발)
```

---

## 환경 변수 요약

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SEG_MODEL` | `/models/seg_model_int8.tflite` | 세그멘테이션 모델 경로 |
| `CLS_MODEL` | `/models/cls_model_int8.tflite` | 분류 모델 경로 |
| `REDIS_HOST`| `redis` | Redis 서비스 이름/IP |
| `REDIS_PORT`| `6379` | Redis 포트 |
| `INTERVAL` | `120` | 캡처 주기(초) – 실기 모드 |
| `NUM_THREADS` | `2` | TFLite 추론 스레드 수 |

환경 변수는 `docker-compose.yml` 에 정의되어 있으며 필요에 따라 오버라이드할 수 있습니다.

---

## 결과 포맷 예시

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

Redis 키(`result:<uuid>`)에 JSON 문자열로 저장되므로, 다음과 같이 간단히 조회할 수 있습니다:

```python
import redis, json
r = redis.Redis(host="REDIS_IP", port=6379)
raw = r.get("result:abc123...")
print(json.loads(raw))
```

---

## 개발 가이드

1. **모델 교체**
   * `models/` 에 새 TFLite 모델 복사 → `docker-compose.yml` 의 환경 변수 수정
2. **카메라 없는 개발**
   * `python vision_app/one_img_processor.py`
3. **컨테이너 재빌드**
   * `docker compose build vision_app`
4. **로그 확인**
   * `docker compose logs -f vision_app`

---

## 기여(Contributing)

Pull Request 환영합니다!  버그 리포트·성능 개선·문서 업데이트 등 어떤 기여든 감사히 검토하겠습니다.

---

## 라이선스

MIT License © 2025 DaisyBum & Contributors
