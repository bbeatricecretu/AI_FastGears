"""python -m fastgears  ->  starts camera + detectors + LLM co-pilot + dashboard."""
import argparse
import logging
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="FastGears driver monitor")
    ap.add_argument("--camera", help="webcam index (0,1,..) or a video file for dev replay")
    ap.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"),
                    help="Windows camera capture backend")
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--no-llm", action="store_true", help="rule-based messages only")
    ap.add_argument("--no-objects", action="store_true", help="skip YOLO phone detection (faster)")
    ap.add_argument("--no-rppg", action="store_true", help="skip camera pulse")
    a = ap.parse_args()
    # CLI overrides env, before config is imported
    if a.camera is not None:
        os.environ["CAMERA"] = a.camera
    if a.camera_backend:
        os.environ["CAMERA_BACKEND"] = a.camera_backend
    if a.port:
        os.environ["PORT"] = str(a.port)
    if a.host:
        os.environ["HOST"] = a.host
    if a.no_llm:
        os.environ["LLM_PROVIDER"] = "none"
    if a.no_objects:
        os.environ["ENABLE_OBJECTS"] = "0"
    if a.no_rppg:
        os.environ["ENABLE_RPPG"] = "0"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn
    from .config import settings
    print(f"\n  FastGears dashboard ->  http://{settings.host}:{settings.port}\n")
    uvicorn.run("fastgears.server:app", host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
