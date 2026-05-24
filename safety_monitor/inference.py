from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

from safety_monitor.models import Detection


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou < thr]
    return keep


class Detector:
    """ONNX YOLO11/YOLOv8 계열(단일 출력) 또는 OpenCV HOG 데모 폴백."""

    def __init__(
        self,
        model_path: str,
        input_size: int,
        conf: float,
        nms: float,
        class_ids_person: Sequence[int],
        use_demo: bool,
    ) -> None:
        self.input_size = int(input_size)
        self.conf = float(conf)
        self.nms = float(nms)
        self.class_ids = set(int(x) for x in class_ids_person)
        self.use_demo = bool(use_demo)
        self._session = None
        self._hog = None

        path = (model_path or "").strip()
        if path and Path(path).is_file() and ort is not None:
            self._session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
        elif not self.use_demo:
            self.use_demo = True

        if self._session is None and self.use_demo:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        if self._session is not None:
            return self._detect_onnx(bgr)
        if self._hog is not None:
            return self._detect_hog(bgr)
        return []

    def _letterbox(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Ultralytics LetterBox(center=True)와 동일: 비율 유지 후 114로 가운데 패딩."""
        h, w = img.shape[:2]
        new_h, new_w = self.input_size, self.input_size
        r = min(new_h / h, new_w / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]
        dw /= 2.0
        dh /= 2.0
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        pad = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        pad_x, pad_y = float(left), float(top)
        return pad, r, (pad_x, pad_y)

    def _yolo_xywh_preds_to_dets(
        self,
        preds: np.ndarray,
        pad_x: float,
        pad_y: float,
        r: float,
        h0: int,
        w0: int,
    ) -> List[Detection]:
        """preds: (num_predictions, 4 + num_classes), xywh는 letterbox 입력(640) 픽셀 기준.

        Ultralytics scale_boxes(xywh=True): (cx,cy)에서 pad 빼고, 네 값 모두 gain으로 나눔.
        """
        boxes_xyxy: List[List[float]] = []
        scores: List[float] = []
        clss: List[int] = []
        gain = float(r)
        for p in preds:
            cx, cy, bw, bh = p[:4]
            rest = p[4:]
            cid = int(np.argmax(rest))
            sc = float(rest[cid])
            if sc < self.conf:
                continue
            if self.class_ids and cid not in self.class_ids:
                continue
            cx0 = (float(cx) - pad_x) / gain
            cy0 = (float(cy) - pad_y) / gain
            bw0 = float(bw) / gain
            bh0 = float(bh) / gain
            x1 = cx0 - bw0 * 0.5
            y1 = cy0 - bh0 * 0.5
            x2 = cx0 + bw0 * 0.5
            y2 = cy0 + bh0 * 0.5
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(sc)
            clss.append(cid)
        dets: List[Detection] = []
        if not boxes_xyxy:
            return dets
        b = np.array(boxes_xyxy, dtype=np.float32)
        s = np.array(scores, dtype=np.float32)
        keep = _nms(b, s, self.nms)
        for i in keep:
            dets.append(
                Detection(
                    float(b[i, 0]),
                    float(b[i, 1]),
                    float(b[i, 2]),
                    float(b[i, 3]),
                    float(s[i]),
                    int(clss[i]),
                )
            )
        return dets

    def _detect_onnx(self, bgr: np.ndarray) -> List[Detection]:
        assert self._session is not None
        h0, w0 = bgr.shape[:2]
        # Ultralytics YOLO8/11 ONNX는 RGB로 학습·보내기
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blob, r, (pad_x, pad_y) = self._letterbox(rgb)
        x = blob.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]

        inp = self._session.get_inputs()[0].name
        out = self._session.run(None, {inp: x})[0]
        out = np.squeeze(out)
        dets: List[Detection] = []

        # Ultralytics export nms=True: [1, K, 6] → xyxy(letterbox 좌표), conf, cls
        if out.ndim == 2 and out.shape[1] == 6:
            gain = float(r)
            for row in out:
                x1, y1, x2, y2, sc, cid = row[:6]
                if sc < self.conf:
                    continue
                if self.class_ids and int(cid) not in self.class_ids:
                    continue
                x1 = (float(x1) - pad_x) / gain
                y1 = (float(y1) - pad_y) / gain
                x2 = (float(x2) - pad_x) / gain
                y2 = (float(y2) - pad_y) / gain
                dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(sc), int(cid)))
        elif out.ndim == 2 and out.shape[0] > 6 and out.shape[1] > out.shape[0]:
            # YOLO11/YOLOv8: [4+nc, num_anchors] (예: 84×8400)
            dets = self._yolo_xywh_preds_to_dets(out.T, pad_x, pad_y, r, h0, w0)
        elif out.ndim == 2 and out.shape[1] > 6 and out.shape[0] > out.shape[1]:
            # 일부보내기: [num_anchors, 4+nc]
            dets = self._yolo_xywh_preds_to_dets(out, pad_x, pad_y, r, h0, w0)
        for d in dets:
            d.x1 = max(0, min(w0 - 1, d.x1))
            d.x2 = max(0, min(w0 - 1, d.x2))
            d.y1 = max(0, min(h0 - 1, d.y1))
            d.y2 = max(0, min(h0 - 1, d.y2))
        return dets

    def _detect_hog(self, bgr: np.ndarray) -> List[Detection]:
        assert self._hog is not None
        rects, weights = self._hog.detectMultiScale(
            bgr, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        dets: List[Detection] = []
        for (x, y, w, h), wgt in zip(rects, weights):
            sc = float(wgt[0]) if hasattr(wgt, "__len__") else float(wgt)
            if sc < self.conf:
                continue
            dets.append(Detection(float(x), float(y), float(x + w), float(y + h), sc, 0))
        if len(dets) > 1:
            b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
            s = np.array([d.score for d in dets], dtype=np.float32)
            keep = _nms(b, s, self.nms)
            dets = [dets[i] for i in keep]
        return dets
