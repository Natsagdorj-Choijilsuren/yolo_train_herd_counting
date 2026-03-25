# Training YOLO for herd counting

## Train

Run training with:

```bash
python -m src.train.train_yolo --config src/config/train_config.yaml --load_type best
```

Options:

- `--config` sets the training config file path
- `--load_type best` loads `best.pt` from stage 1 for stage 2
- `--load_type last` loads `last.pt` from stage 1 for stage 2

## Export TFLite

Export the trained stage 2 weights to TFLite with:

```bash
python -m src.export.export_tflite --stage2-name experiment_stage2 --imgsz 640 --weights best
```

Options:

- `--weights best` exports `best.pt`
- `--weights last` exports `last.pt`
- `--int8` exports INT8 TFLite
- `--half` exports FP16 TFLite

Example with INT8:

```bash
python -m src.export.export_tflite --stage2-name experiment_stage2 --imgsz 640 --weights best --int8
```
