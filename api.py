from fastapi import FastAPI, UploadFile, File, Query, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel, validator
from ultralytics import YOLO
import shutil, os, base64, uuid, asyncio
import numpy as np
import cv2
import time
from datetime import datetime
from typing import Union

# -----------------------------------
# APP
# -----------------------------------
app = FastAPI(title="VisionQA API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# MODEL
# -----------------------------------
model = YOLO("model/best.pt")

# -----------------------------------
# ROUTER
# -----------------------------------
router = APIRouter(prefix="/api", tags=["Vision"])


# ===================================
# ROOT & HEALTH
# ===================================
@router.get("/")
def root():
    return {"message": "VisionQA API running", "docs": "/docs"}


@router.get("/health")
def health():
    return {"status": "ok", "version": "4.0"}


# ===================================
# SHARED: core detection logic
# ===================================
async def run_detection(temp_path: str, conf: float, iou: float, lightweight: bool):
    results = await asyncio.to_thread(
        model.predict, source=temp_path, conf=conf, iou=iou
    )

    detections    = []
    annotated_b64 = None

    for r in results:
        if not lightweight:
            ok, enc = cv2.imencode(".jpg", r.plot())
            if ok:
                annotated_b64 = base64.b64encode(enc).decode()

        if r.boxes is not None:
            for box in r.boxes:
                cls_id       = int(box.cls)
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class":      model.names[cls_id],
                    "confidence": round(float(box.conf), 3),
                    "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

    response = {
        "status":           "OK",
        "total_detections": len(detections),
        "detections":       detections,
        "has_defect":       len(detections) > 0
    }
    if not lightweight:
        response["annotated_image"] = annotated_b64

    return response


# ===================================
# /detect  — multipart file upload
# ===================================
@router.post("/detect")
async def detect(
    image:       UploadFile = File(...),
    conf:        float      = Query(0.35),
    iou:         float      = Query(0.50),
    lightweight: bool       = Query(False),
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(image.file, buf)

    try:
        return await run_detection(temp_path, conf, iou, lightweight)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ===================================
# /detect-b64  — JSON + base64 image
# ===================================
class Base64ImagePayload(BaseModel):
    image:       str
    conf:        float = 0.35
    iou:         float = 0.50
    lightweight: bool  = False


@router.post("/detect-b64")
async def detect_b64(payload: Base64ImagePayload, background_tasks: BackgroundTasks):

    # decode image
    img_bytes = base64.b64decode(payload.image)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(temp_path, frame)

    try:
        result = await run_detection(temp_path, payload.conf, payload.iou, payload.lightweight)

        # AUTO LOGGING
        if result["has_defect"]:
            background_tasks.add_task(auto_log, "FAIL", "true")
        else:
            background_tasks.add_task(auto_log, "PASS", "false")

        return result

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ===================================
# /save-defect — multipart
# FIX 2: folder renamed to inspection_outputs (consistent with camera_agent)
# FIX 3: millisecond timestamp to avoid filename collisions
# ===================================
@app.post("/api/save-defect")
async def save_defect(image: UploadFile):

    # FIX 2: use "inspection_outputs" — consistent folder name everywhere
    os.makedirs("inspection_outputs", exist_ok=True)

    # FIX 3: millisecond timestamp prevents collision when frames arrive quickly
    filename = f"{int(time.time() * 1000)}_{image.filename}"

    file_path = os.path.join("inspection_outputs", filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await image.read())

    print(f"[SAVE] Image saved → {file_path}")

    return {"path": file_path}


# ===================================
# /save-defect-b64 — base64 JSON
# FIX 2: folder renamed to inspection_outputs
# ===================================
class SaveDefectPayload(BaseModel):
    image:      str
    defects:    str = "unknown"
    confidence: str = "—"
    timestamp:  str = ""


@router.post("/save-defect-b64")
async def save_defect_b64(payload: SaveDefectPayload):
    try:
        img_bytes = base64.b64decode(payload.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # FIX 2: consistent folder name
    folder   = "inspection_outputs"
    os.makedirs(folder, exist_ok=True)
    ts       = payload.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"defect_{ts}.jpg"
    path     = os.path.join(folder, filename)
    cv2.imwrite(path, frame)

    return {"status": "saved", "path": path}


# ===================================
# /alert — trigger alert
# ===================================
class AlertPayload(BaseModel):
    defects:    str = "unknown"
    confidence: str = "—"
    timestamp:  str = ""


@router.post("/alert")
async def trigger_alert(payload: AlertPayload):
    ts  = payload.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"DEFECT DETECTED | "
        f"defects={payload.defects} | "
        f"conf={payload.confidence} | "
        f"ts={ts}"
    )
    print(f"🚨 {msg}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/alert_log.txt", "a") as f:
        f.write(msg + "\n")

    return {"status": "alert triggered", "message": msg}


# ===================================
# /log — log result (TRUE + FALSE branch)
#
# conf and defects accept both str and number from Langflow
#
# Langflow HTTP Request Node config:
#   URL    : http://127.0.0.1:8000/api/log
#   Method : POST
#   Headers: Content-Type: application/json
#   Body table:
#     status  = PASS or FAIL
#     branch  = true or false
#     conf    = 0.95  (number is fine)
#     defects = crack (or 0 for no defect)
# ===================================
class LogPayload(BaseModel):
    status:  str                    = "UNKNOWN"
    branch:  str                    = "false"
    conf:    Union[str, float, int] = "—"
    defects: Union[str, float, int] = "None"
    ts:      str                    = ""

    # Convert conf and defects to string automatically
    @validator("conf", "defects", pre=True)
    def coerce_to_str(cls, v):
        return str(v)


@router.post("/log")
async def log_result(data: LogPayload):
    os.makedirs("outputs", exist_ok=True)
    ts   = data.ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts} | status={data.status} | branch={data.branch} | "
        f"conf={data.conf} | defects={data.defects}\n"
    )
    with open("outputs/log.txt", "a") as f:
        f.write(line)

    print(f"📝 Logged: {line.strip()}")
    return {"status": "logged", "line": line.strip()}


# ===================================
# DEBUG — list all registered routes
# ===================================
@router.get("/routes")
def list_routes():
    return [
        {"path": r.path, "methods": list(r.methods)}
        for r in app.routes if hasattr(r, "methods")
    ]


# ===================================
# BACKGROUND TASK: auto_log
# ===================================
def auto_log(status: str, branch: str):
    time.sleep(0.1)
    line = f"{datetime.now()} | status={status} | branch={branch}\n"
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/log.txt", "a") as f:
        f.write(line)


# ===================================
# MOUNT ROUTER
# ===================================
app.include_router(router)
