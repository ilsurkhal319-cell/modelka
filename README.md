# Распознавание дефектов

Основная модель: `model_training/handoff/expanded_v2/weights/best.pt`.
Классы: коррозия, трещины труб, утечки труб, повреждение изоляции кабеля, дефекты крепежа.

```bash
python -m pip install -r model_training/requirements.txt
python predict.py /путь/к/фото.jpg
```

Можно передать папку изображений, видео или `0` для камеры.
По умолчанию используется CPU. Для NVIDIA GPU добавьте `--device 0`.
Порог уверенности: `--conf 0.35`. Результаты с рамками и текстовые файлы
предсказаний сохраняются в `runs/predict/`.

Инструкции по обучению: [model_training/README.md](model_training/README.md).
