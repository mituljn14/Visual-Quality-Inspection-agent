def format_detections(results):
    detections = []

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                detections.append({
                    "class": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()
                })

    return detections


def detections_to_text(detections):
    if not detections:
        return "No defects detected."

    text = "Detected defects:\n"

    for d in detections:
        text += f"- Class {d['class']} with confidence {d['confidence']:.2f}\n"

    return text