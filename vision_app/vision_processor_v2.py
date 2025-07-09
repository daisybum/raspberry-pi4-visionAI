from __future__ import annotations

"""
vision_processor_v2.py

이미지 파일과 센서 JSON 파일을 동시에 입력받아
세그멘테이션(.tflite) 및 이미지 분류(.tflite) 모델에 대한 추론을 실행한다.

사용 예)
---------
$ python vision_processor_v2.py \
    --image data/example.jpg \
    --sensor_json data/example.json \
    --seg_model models/seg_model_int8.tflite \
    --cls_model models/cls_model_int8.tflite \
    --output_dir outputs --save_mask
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np  # type: ignore
from PIL import Image  # type: ignore

# tflite-runtime 우선, 없으면 tensorflow-lite로 폴백
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:
    import tensorflow as tf  # type: ignore

    tflite = tf.lite  # pyright: ignore

# --------------------------------------------------------------------------------------
# 설정 & 로거
# --------------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("vision_processor_v2")

NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))

# --------------------------------------------------------------------------------------
# Sensor helpers
# --------------------------------------------------------------------------------------

def _sensor_to_vec(sensor_data: dict | None) -> np.ndarray:
    """센서 JSON → (1, 6) float32 벡터 변환"""

    if not sensor_data:
        return np.zeros((1, 6), dtype=np.float32)

    keys = [
        "objectTemp",
        "humi",
        "pressure",
        "latitude",
        "longitude",
        "height",
    ]
    vec = [float(sensor_data.get(k, 0.0)) for k in keys]
    return np.asarray([vec], dtype=np.float32)

# --------------------------------------------------------------------------------------
# Delegate & Interpreter helpers
# --------------------------------------------------------------------------------------

def _load_delegate(delegate_name: str | None):
    """EdgeTPU 등 하드웨어 가속용 delegate 로드."""

    if delegate_name is None:
        return None

    delegate_name = delegate_name.lower()
    if delegate_name == "edgetpu":
        try:
            return tflite.load_delegate("libedgetpu.so.1")
        except ValueError as e:  # 라이브러리 누락
            logger.warning("EdgeTPU delegate 로드 실패: %s", e)
            return None
    logger.warning("알 수 없는 delegate: %s", delegate_name)
    return None


def _new_interpreter(model_path: Path, delegate: Any | None = None):
    """주어진 모델 경로로 Interpreter 인스턴스를 생성한다."""

    kwargs: Dict[str, Any] = {"model_path": str(model_path), "num_threads": NUM_THREADS}
    if delegate is not None:
        kwargs["experimental_delegates"] = [delegate]
    return tflite.Interpreter(**kwargs)

# --------------------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------------------

def _prepare_input(pil: Image.Image, target_hw: Tuple[int, int], input_info: Dict[str, Any]) -> np.ndarray:
    """PIL 이미지를 모델 입력 요건(dtype, 정규화)에 맞게 ndarray로 변환."""

    w, h = target_hw
    img = pil.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img)

    dtype = input_info["dtype"]
    if dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        arr = arr.astype(np.uint8)
    elif dtype == np.int8:
        # int8 대칭 양자화 처리
        scale, zero_point = input_info.get("quantization", (1.0, 0))
        if scale == 0:
            scale = 1.0 / 127.0
        arr = ((arr.astype(np.float32) / 255.0) / scale + zero_point).astype(np.int8)
    else:
        raise ValueError(f"지원되지 않는 입력 dtype: {dtype}")

    return np.expand_dims(arr, axis=0)

# --------------------------------------------------------------------------------------
# Inference routine
# --------------------------------------------------------------------------------------

def process_single(
    image_path: Path,
    sensor_json_path: Path,
    seg_interp: Any,
    cls_interp: Any | None = None,
):
    """하나의 이미지 + 센서 JSON에 대해 추론을 실행하고 결과 dict 반환."""

    # 센서 JSON → 벡터
    try:
        sensor_raw = json.loads(sensor_json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("센서 JSON 파일을 찾을 수 없습니다: %s", sensor_json_path)
        sensor_raw = None
    sensor_vec = _sensor_to_vec(sensor_raw)

    # 이미지 로드
    ts0 = time.time()
    with Image.open(image_path) as pil:
        pil = pil.convert("RGB")

        # ------------------------------
        # Segmentation
        # ------------------------------
        seg_inputs = seg_interp.get_input_details()

        # 입력 텐서 구분: 이미지 입력은 4D (N,H,W,C) 또는 3D (H,W,C)
        img_in   = next((inp for inp in seg_inputs if len(inp["shape"]) >= 3), None)
        sensor_in = next((inp for inp in seg_inputs if len(inp["shape"]) == 2), None)

        if img_in is None:
            raise RuntimeError("세그멘테이션 모델에서 이미지 입력 텐서를 찾을 수 없습니다.")

        seg_size = (
            img_in["shape"][2] if len(img_in["shape"]) >= 3 else img_in["shape"][1],
            img_in["shape"][1] if len(img_in["shape"]) >= 3 else img_in["shape"][0],
        )
        seg_arr = _prepare_input(pil, seg_size, img_in)
        seg_interp.set_tensor(img_in["index"], seg_arr)

        # 센서 입력(있는 경우)
        if sensor_in is not None:
            sensor_arr_use = sensor_vec.astype(sensor_in["dtype"])
            if sensor_arr_use.shape != tuple(sensor_in["shape"]):
                sensor_arr_use = sensor_arr_use.reshape(sensor_in["shape"])
            seg_interp.set_tensor(sensor_in["index"], sensor_arr_use)

        seg_interp.invoke()
        seg_out = seg_interp.get_output_details()[0]
        mask = seg_interp.get_tensor(seg_out["index"])[0]
        if mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1).astype(np.uint8)
        else:
            if mask.dtype != np.float32:
                mask = mask.astype(np.float32)
            mask = (mask > 0.0).astype(np.uint8)

        # ------------------------------
        # Classification (선택)
        # ------------------------------
        cls_pred = None
        if cls_interp is not None:
            cls_in = cls_interp.get_input_details()[0]
            cls_size = (cls_in["shape"][2], cls_in["shape"][1])
            cls_arr = _prepare_input(pil, cls_size, cls_in)
            cls_interp.set_tensor(cls_in["index"], cls_arr)
            cls_interp.invoke()
            cls_out = cls_interp.get_output_details()[0]
            cls_pred = cls_interp.get_tensor(cls_out["index"])[0]

    return {
        "mask": mask,
        "cls_pred": cls_pred,
        "elapsed": time.time() - ts0,
    }

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vision Processor v2 – 이미지 + 센서 JSON 동시 입력")
    p.add_argument("--image", default="data/20220428_000_40P0S1R1AX_0_20220702_065412.jpg", help="입력 이미지 파일 경로")
    p.add_argument("--sensor_json", default="data/20220428_000_40P0S1R1AX_0_20220702_065412.json", help="센서 JSON 파일 경로")
    p.add_argument("--seg_model", default="/models/seg_model_sensor_int8.tflite", help="세그멘테이션 .tflite 경로")
    p.add_argument("--cls_model", default="/models/cls_model_int8.tflite", help="분류 .tflite 경로 (선택)")
    p.add_argument("--delegate", choices=["edgetpu"], default=None, help="사용 delegate")
    p.add_argument("--output_dir", default="v2_output", help="결과 저장 폴더")
    p.add_argument("--save_mask", action="store_true", help="세그멘테이션 마스크 PNG 저장 여부")
    return p


def main():
    args = _build_argparser().parse_args()

    img_path = Path(args.image)
    sensor_json_path = Path(args.sensor_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delegate = _load_delegate(args.delegate)
    seg_interp = _new_interpreter(Path(args.seg_model), delegate)
    seg_interp.allocate_tensors()
    cls_interp = None
    if args.cls_model:
        cls_interp = _new_interpreter(Path(args.cls_model), delegate)
        cls_interp.allocate_tensors()

    result = process_single(img_path, sensor_json_path, seg_interp, cls_interp)
    logger.info("처리 완료 – %.3f s", result["elapsed"])

    if args.save_mask:
        mask_img = Image.fromarray(result["mask"])
        mask_img.save(out_dir / f"{img_path.stem}_mask.png")
        logger.info("마스크 PNG 저장 완료")

    if result["cls_pred"] is not None:
        logger.info("분류 logits: %s", np.array2string(result["cls_pred"], precision=3, separator=", "))


if __name__ == "__main__":
    main()
