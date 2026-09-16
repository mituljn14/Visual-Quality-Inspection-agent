def run_detection(model, image_path):
    results = model.predict(
        source=image_path,
        conf=0.35,
        iou=0.5,
        imgsz=640
    )
    return results