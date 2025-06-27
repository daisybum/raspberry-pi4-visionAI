from __future__ import annotations

"""
Vision Processor (no-camera version)

✓ 세그멘테이션 + 분류 모델 동시 실행
✓ Redis로 결과 전송
✓ ./data/example.jpg를 2분마다 추론
✓ 2025-05-21 리팩터링 포인트 반영
✓ ────────────── 시각화 기능 추가 ──────────────
"""

import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

# ─── 새로 추가된 의존성 ─────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# ------------------------------------------------

# ─── 로깅 설정 ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VISION] %(levelname)s: %(message)s",
)
logger = logging.getLogger("vision_processor")

# ─── 환경 변수 & 상수 ─────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SEG_MODEL_PATH = Path(os.getenv("SEG_MODEL", "/models/seg_model_int8.tflite"))
CLS_MODEL_PATH = Path(os.getenv("CLS_MODEL", "/models/cls_model_int8.tflite"))

CAPTURE_INTERVAL = 2           # ★ 2분
SAMPLE_IMG_PATH = Path("./data/example.jpg")  # ★ 고정 입력 이미지
NUM_THREADS = int(os.getenv("NUM_THREADS", "2"))

# ─── (시각화용) 팔레트 & 클래스명 ─────────────────────────────────────────
#  필요 시 교체하세요.
PALETTE = np.array(
    [
        (0, 0, 0),       # background
        (255, 0, 0),     # class 1
        (0, 255, 0),     # class 2
        (0, 0, 255),     # class 3
        (255, 255, 0),   # class 4 ...
    ],
    dtype=np.uint8,
)
CLASS_NAMES = [f"class_{i}" for i in range(len(PALETTE))]

# ─── Delegate 헬퍼 ────────────────────────────────────────────────────────
def _load_xnnpack_delegate() -> Optional[Any]:
    for lib in ("libxnnpack.so", "libtensorflowlite_delegate_xnnpack.so"):
        try:
            return load_delegate(lib)
        except Exception:
            continue
    return None

# ─── 모델 로드 ───────────────────────────────────────────────────────────
def load_models() -> Tuple[Interpreter, Interpreter]:
    delegate = _load_xnnpack_delegate()

    def _new(model: Path) -> Interpreter:
        kw = {"model_path": str(model), "num_threads": NUM_THREADS}
        if delegate is not None:
            kw["experimental_delegates"] = [delegate]
        return Interpreter(**kw)

    logger.info(f"Loading segmentation model → {SEG_MODEL_PATH}")
    seg = _new(SEG_MODEL_PATH)
    seg.allocate_tensors()

    logger.info(f"Loading classification model → {CLS_MODEL_PATH}")
    cls = _new(CLS_MODEL_PATH)
    cls.allocate_tensors()

    return seg, cls

# ─── 전처리 유틸 ──────────────────────────────────────────────────────────
def _prepare_input(img: Image.Image, target_hw: Tuple[int, int], details: Dict) -> np.ndarray:
    w, h = target_hw
    arr = np.asarray(img.resize((w, h))).astype(np.float32) / 255.0

    dtype = details["dtype"]
    if dtype == np.uint8:
        arr = (arr * 255).astype(np.uint8)
    elif dtype == np.int8:
        scale, zero = details["quantization"]
        arr = ((arr / scale) + zero).astype(np.int8)
    return np.expand_dims(arr, 0)

# ─── 시각화 함수들 ────────────────────────────────────────────────────────
def create_legend_patches(
    palette: np.ndarray | List[Tuple[int, int, int]],
    class_names: List[str],
):
    """팔레트·클래스명으로 matplotlib Patch 리스트 생성."""
    patches: List[mpatches.Patch] = []
    for idx, (r, g, b) in enumerate(palette):
        patches.append(
            mpatches.Patch(
                color=(r / 255, g / 255, b / 255),
                label=class_names[idx],
            )
        )
    return patches


def visualize_and_save(  # save_image 기본값 False!
    file_name: str,
    orig: np.ndarray,
    color_mask: np.ndarray,
    overlay: np.ndarray,
    legend_patches: List[mpatches.Patch],
    output_dir: str = "./vis",
    save_image: bool = False,
) -> str:
    """원본/마스크/오버레이 3분할 시각화(저장 안 함)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Segmentation – {file_name}", fontsize=16)

    titles = ["Original", "Mask", "Overlay"]
    images = [orig, color_mask, overlay]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=12,
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    if save_image:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir, f"{os.path.splitext(file_name)[0]}_visual.png"
        )
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    else:
        plt.show(block=False)
        plt.pause(0.001)  # 짧게 띄워두기 (CLI 환경에선 무시)
        plt.close(fig)
        return "[Visualization shown only]"

# ─── 이미지 추론 ──────────────────────────────────────────────────────────
def process_image(
    pil: Image.Image,
    seg_interp: Interpreter,
    cls_interp: Interpreter,
) -> Dict[str, Any]:
    ts0 = time.time()
    try:
        pil = pil.convert("RGB")

        # 세그멘테이션
        seg_start = time.time()
        seg_in = seg_interp.get_input_details()[0]
        seg_arr = _prepare_input(pil, (seg_in["shape"][2], seg_in["shape"][1]), seg_in)
        seg_interp.set_tensor(seg_in["index"], seg_arr)
        seg_interp.invoke()
        seg_out = seg_interp.get_output_details()[0]
        mask = seg_interp.get_tensor(seg_out["index"])[0]
        seg_elapsed_ms = (time.time() - seg_start) * 1000
        mask = np.argmax(mask, -1) if mask.shape[-1] > 1 else (mask > 0).astype(np.uint8)

        # 분류
        cls_start = time.time()
        cls_in = cls_interp.get_input_details()[0]
        cls_arr = _prepare_input(pil, (cls_in["shape"][2], cls_in["shape"][1]), cls_in)
        cls_interp.set_tensor(cls_in["index"], cls_arr)
        cls_interp.invoke()
        cls_out = cls_interp.get_output_details()[0]
        scores = cls_interp.get_tensor(cls_out["index"])[0]
        cls_elapsed_ms = (time.time() - cls_start) * 1000
        top_idx = int(scores.argmax())
        top_score = float(scores[top_idx])

        # ── 시각화 준비 ────────────────────────────────────────────────
        # ── 시각화 준비 ────────────────────────────────────────────────
        orig_np = np.asarray(pil)

        # 1) 모델 출력 마스크(예: 513×513)를 원본 해상도로 리사이즈
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (orig_np.shape[1], orig_np.shape[0]),  # (W, H)
                resample=Image.NEAREST,               # 레이블 보존
            )
        )

        # 2) 컬러 마스크 & 오버레이 생성
        color_mask = PALETTE[mask_resized]                 # (H, W, 3)
        overlay = (0.4 * orig_np + 0.6 * color_mask).astype(np.uint8)

        legend = create_legend_patches(PALETTE, CLASS_NAMES)
        visualize_and_save(
            file_name=Path(SAMPLE_IMG_PATH).name,
            orig=orig_np,
            color_mask=color_mask,
            overlay=overlay,
            legend_patches=legend,
            save_image=False,                         # ← 저장 안 함
        )

        return {
            "segmentation": {
                "mask_shape": mask.shape,
                "unique_labels": np.unique(mask).tolist(),
            },
            "classification": {"id": top_idx, "score": top_score},
            "seg_inference_time_ms": seg_elapsed_ms,
            "cls_inference_time_ms": cls_elapsed_ms,
            "inference_time_ms": int((time.time() - ts0) * 1000),
        }
    except Exception as exc:
        logger.exception("추론 실패")
        return {"error": str(exc), "inference_time_ms": int((time.time() - ts0) * 1000)}

# ─── Redis 저장 ───────────────────────────────────────────────────────────
def store_result(r: redis.Redis, result: Dict[str, Any]) -> Optional[str]:
    key = f"result:{uuid.uuid4().hex}"
    try:
        r.set(key, json.dumps(result, ensure_ascii=False), ex=3600)
        return key
    except redis.RedisError:
        logger.exception("Redis 저장 실패")
        return None

# ─── 메인 루프 ────────────────────────────────────────────────────────────
def main() -> None:
    logger.info(f"Vision processor 시작 (주기 {CAPTURE_INTERVAL}s, sample={SAMPLE_IMG_PATH})")
    if not SAMPLE_IMG_PATH.exists():
        raise FileNotFoundError(f"{SAMPLE_IMG_PATH} not found")

    seg, cls = load_models()

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        logger.info(f"Redis 연결 성공 → {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError:
        logger.exception("Redis 연결 실패·비활성화 모드")
        r = None

    while True:
        try:
            with Image.open(SAMPLE_IMG_PATH) as pil:  # ★ 고정 이미지 로드
                result = process_image(pil, seg, cls)

            if r is not None:
                key = store_result(r, result)
                if key:
                    logger.info(f"결과 Redis 저장 → {key}")

            logger.info(
                "Seg labels=%s | Seg=%.1f ms | Cls=(%d, %.3f, %.1f ms) | Total=%.1f ms",
                result.get("segmentation", {}).get("unique_labels"),
                result.get("seg_inference_time_ms", 0),
                result.get("classification", {}).get("id", -1),
                result.get("classification", {}).get("score", 0.0),
                result.get("cls_inference_time_ms", 0),
                result.get("inference_time_ms", 0),
            )
        except KeyboardInterrupt:
            logger.info("종료 신호 수신·중단")
            break
        except Exception:
            logger.exception("메인 루프 예외")
        time.sleep(CAPTURE_INTERVAL)  # ★ 2분 대기

if __name__ == "__main__":
    main()
