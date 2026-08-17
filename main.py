from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("runs/detect/train/weights/best.pt")


class ObjectDetector:
    """YOLO object detector wrapper for image inference."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Train the model first or update MODEL_PATH in app.py."
            )

        self.model = YOLO(str(self.model_path))

    def predict(self, image: Union[Image.Image, np.ndarray], conf: float = 0.25):
        """Run YOLO inference and return annotated image + detections."""
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image

        results = self.model.predict(
            source=image_np,
            conf=conf,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()
        annotated = Image.fromarray(annotated[..., ::-1])

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id],
                    "confidence": round(confidence, 4),
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                })

        return annotated, detections


def detect_image(
    image: Union[Image.Image, np.ndarray],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    conf: float = 0.25,
):
    """Simple function interface for image detection."""
    detector = ObjectDetector(model_path)
    return detector.predict(image, conf=conf)
