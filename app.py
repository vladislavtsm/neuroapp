from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - depends on local environment
    Image = ImageDraw = ImageFont = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
REPORT_DIR = BASE_DIR / "reports"
DB_PATH = BASE_DIR / "history.db"
MODEL_PATH = Path(os.getenv("YOLO_MODEL", BASE_DIR / "models" / "yolov8n.pt"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))
TARGET_CLASSES = {
    item.strip().lower()
    for item in os.getenv(
        "TARGET_CLASSES",
        "scooter,electric scooter,kick scooter,bicycle,motorcycle",
    ).split(",")
    if item.strip()
}

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024


@dataclass
class DetectionResult:
    count: int
    labels: list[dict[str, Any]]
    model: str
    confidence_avg: float
    result_filename: str


def is_target_class(class_name: str) -> bool:
    normalized = class_name.strip().lower()
    return normalized in TARGET_CLASSES


def display_label(class_name: str) -> str:
    normalized = class_name.strip().lower()
    if "scooter" in normalized:
        return class_name
    if normalized in {"bicycle", "motorcycle"}:
        return f"possible_scooter ({class_name})"
    return class_name


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)


def init_db() -> None:
    ensure_directories()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                mode TEXT NOT NULL,
                count INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                result_image TEXT NOT NULL
            )
            """
        )
        conn.commit()


def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS


def media_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return "video" if ext in ALLOWED_VIDEO_EXTENSIONS else "image"


def unique_filename(original: str) -> str:
    safe = secure_filename(original) or "media"
    stem = Path(safe).stem[:60]
    suffix = Path(safe).suffix.lower()
    return f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"


def read_video_preview(path: Path):
    if cv2 is None:
        raise RuntimeError("Для обработки видео требуется пакет opencv-python.")

    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("Не удалось прочитать первый кадр видео.")
    return frame


def read_image(path: Path):
    if cv2 is None:
        if Image is None:
            raise RuntimeError("Для обработки изображений установите opencv-python или Pillow.")
        image = Image.open(path).convert("RGB")
        return image

    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError("Не удалось прочитать изображение.")
    return frame


class VisionProcessor:
    def __init__(self) -> None:
        self._model = None
        self._model_error = None

    def _load_yolo(self):
        if self._model is not None:
            return self._model
        if not MODEL_PATH.exists():
            return None
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(MODEL_PATH))
            return self._model
        except Exception as exc:  # pragma: no cover - runtime dependency
            self._model_error = str(exc)
            return None

    def process(self, path: Path, media_kind: str) -> DetectionResult:
        frame = read_video_preview(path) if media_kind == "video" else read_image(path)
        result_filename = f"{path.stem}_result.jpg"
        result_path = RESULT_DIR / result_filename

        yolo = self._load_yolo()
        if yolo is not None and cv2 is not None:
            return self._process_with_yolo(yolo, frame, result_path, result_filename)

        return self._process_with_heuristics(frame, result_path, result_filename)

    def _process_with_yolo(self, yolo, frame, result_path: Path, result_filename: str) -> DetectionResult:
        results = yolo(frame, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, verbose=False)
        rendered = frame.copy()
        labels: list[dict[str, Any]] = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names.get(class_id, f"class_{class_id}")
            if not is_target_class(class_name):
                continue

            confidence = float(box.conf[0])
            xyxy = [int(round(float(value))) for value in box.xyxy[0].tolist()]
            label = display_label(class_name)
            labels.append(
                {
                    "label": label,
                    "source_label": class_name,
                    "confidence": round(confidence, 3),
                    "box": xyxy,
                }
            )
            self._draw_detection(rendered, xyxy, label, confidence)

        cv2.imwrite(str(result_path), rendered)
        confidence_avg = round(sum(item["confidence"] for item in labels) / len(labels), 3) if labels else 0.0
        model_name = f"YOLOv8 ({MODEL_PATH.name}, conf={YOLO_CONF}, imgsz={YOLO_IMGSZ})"
        return DetectionResult(len(labels), labels, model_name, confidence_avg, result_filename)

    def _process_with_heuristics(self, frame, result_path: Path, result_filename: str) -> DetectionResult:
        if cv2 is None:
            return self._process_with_pillow(frame, result_path, result_filename)

        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (min(width, 1280), int(height * min(width, 1280) / width)))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_x = width / resized.shape[1]
        scale_y = height / resized.shape[0]
        candidates: list[dict[str, Any]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 900 or area > resized.shape[0] * resized.shape[1] * 0.35:
                continue
            aspect = w / max(h, 1)
            if 0.35 <= aspect <= 3.8:
                box = [int(x * scale_x), int(y * scale_y), int((x + w) * scale_x), int((y + h) * scale_y)]
                confidence = min(0.92, max(0.36, area / (resized.shape[0] * resized.shape[1]) * 7))
                candidates.append({"label": "possible_scooter", "confidence": round(confidence, 3), "box": box})

        candidates = sorted(candidates, key=lambda item: item["confidence"], reverse=True)[:12]

        annotated = frame.copy()
        for item in candidates:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (28, 119, 195), 3)
            cv2.putText(
                annotated,
                f"{item['label']} {item['confidence']:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (28, 119, 195),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(result_path), annotated)
        count = len(candidates)
        confidence_avg = round(sum(item["confidence"] for item in candidates) / len(candidates), 3) if candidates else 0.0
        return DetectionResult(count, candidates, "OpenCV heuristic fallback", confidence_avg, result_filename)

    @staticmethod
    def _draw_detection(image, box: list[int], label: str, confidence: float) -> None:
        x1, y1, x2, y2 = box
        color = (28, 119, 195)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        caption = f"{label} {confidence:.2f}"
        text_size, baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        top = max(0, y1 - text_size[1] - baseline - 8)
        right = min(image.shape[1] - 1, x1 + text_size[0] + 12)
        cv2.rectangle(image, (x1, top), (right, top + text_size[1] + baseline + 8), color, -1)
        cv2.putText(
            image,
            caption,
            (x1 + 6, top + text_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_banner(image, text: str) -> None:
        cv2.rectangle(image, (18, 18), (min(image.shape[1] - 18, 560), 72), (18, 34, 48), -1)
        cv2.putText(image, text, (34, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def _process_with_pillow(self, image, result_path: Path, result_filename: str) -> DetectionResult:
        width, height = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        box = [width // 5, height // 4, width * 4 // 5, height * 3 // 4]
        label = "demo_detection"

        draw.rectangle(box, outline=(28, 119, 195, 255), width=4)
        draw.text((box[0], max(8, box[1] - 24)), label, fill=(28, 119, 195, 255))

        image.save(result_path, "JPEG", quality=92)
        labels = [{"label": label, "confidence": 0.5, "box": box}]
        return DetectionResult(1, labels, "Pillow demo fallback", 0.5, result_filename)


processor = VisionProcessor()


def save_history(filename: str, kind: str, mode: str, result: DetectionResult) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO requests (timestamp, filename, media_type, mode, count, result_json, result_image)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                filename,
                kind,
                mode,
                result.count,
                json.dumps(result.labels, ensure_ascii=False),
                result.result_filename,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def load_history(limit: int = 50) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, timestamp, filename, media_type, mode, count, result_json, result_image
            FROM requests
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history = []
    for row in rows:
        item = dict(row)
        item["result"] = json.loads(item.pop("result_json"))
        item["result_url"] = f"/static/results/{item['result_image']}"
        history.append(item)
    return history


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "opencv": cv2 is not None,
            "pillow": Image is not None,
            "model_path": str(MODEL_PATH),
            "model_available": MODEL_PATH.exists(),
            "target_classes": sorted(TARGET_CLASSES),
            "yolo_conf": YOLO_CONF,
            "yolo_iou": YOLO_IOU,
            "yolo_imgsz": YOLO_IMGSZ,
        }
    )


@app.post("/api/process")
def process_media():
    init_db()
    uploaded = request.files.get("media")
    mode = "detection"
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "Загрузите изображение или видео."}), 400
    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Поддерживаются изображения JPG/PNG/BMP/WEBP и видео MP4/AVI/MOV/MKV/WEBM."}), 400

    stored_name = unique_filename(uploaded.filename)
    stored_path = UPLOAD_DIR / stored_name
    uploaded.save(stored_path)
    kind = media_type(stored_name)

    try:
        result = processor.process(stored_path, kind)
        request_id = save_history(stored_name, kind, mode, result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "id": request_id,
            "count": result.count,
            "labels": result.labels,
            "model": result.model,
            "confidence_avg": result.confidence_avg,
            "result_url": f"/static/results/{result.result_filename}",
        }
    )


@app.get("/api/history")
def history():
    init_db()
    return jsonify(load_history())


@app.get("/reports/excel")
def report_excel():
    init_db()
    try:
        from openpyxl import Workbook
    except ImportError:
        return jsonify({"error": "Для Excel-отчёта установите openpyxl."}), 500

    rows = load_history(500)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "История"
    sheet.append(["ID", "Дата", "Файл", "Тип", "Режим", "Количество", "Результат"])
    for item in reversed(rows):
        sheet.append(
            [
                item["id"],
                item["timestamp"],
                item["filename"],
                item["media_type"],
                item["mode"],
                item["count"],
                json.dumps(item["result"], ensure_ascii=False),
            ]
        )

    report_path = REPORT_DIR / f"neuroapp_report_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    workbook.save(report_path)
    return send_file(report_path, as_attachment=True, download_name=report_path.name)


@app.get("/reports/pdf")
def report_pdf():
    init_db()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError:
        return jsonify({"error": "Для PDF-отчёта установите reportlab."}), 500

    rows = load_history(200)
    report_path = REPORT_DIR / f"neuroapp_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    page = canvas.Canvas(str(report_path), pagesize=A4)
    width, height = A4

    font_name = "Helvetica"
    for font_path in (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("AppFont", str(font_path)))
            font_name = "AppFont"
            break

    y = height - 42
    page.setFont(font_name, 16)
    page.drawString(42, y, "Отчёт NeuroApp")
    y -= 32
    page.setFont(font_name, 10)

    for item in rows:
        line = (
            f"#{item['id']} | {item['timestamp']} | {item['mode']} | "
            f"{item['filename']} | объектов: {item['count']}"
        )
        page.drawString(42, y, line[:110])
        y -= 18
        if y < 48:
            page.showPage()
            page.setFont(font_name, 10)
            y = height - 42

    page.save()
    return send_file(report_path, as_attachment=True, download_name=report_path.name)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
