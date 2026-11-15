"""Распознавание дефектов основной моделью expanded_v2."""
from pathlib import Path
import argparse

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "model_training/handoff/expanded_v2/weights/best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Фото, папка, видео или номер камеры (0)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="cpu", help="cpu или 0 для NVIDIA GPU")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Веса не найдены: {args.weights}")
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    source = int(args.source) if args.source.isdecimal() else args.source
    results = model.predict(source=source, conf=args.conf, imgsz=640,
                            device=args.device, save=True, save_txt=True,
                            save_conf=True, stream=True,
                            project=str(Path(__file__).resolve().parent / "runs/predict"))
    for result in results:
        print(f"{result.path}: {result.verbose()}")


if __name__ == "__main__":
    main()
