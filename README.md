# FloodRoad-SAM3

Colab-friendly code for flooded-road segmentation experiments on SpaceNet 8.

The project implements the four experiment configurations from the plan:

- `deeplab`: DeepLabV3+ style supervised baseline using `torchvision.models.segmentation.deeplabv3_resnet50`.
- `sam_text`: SAM3 text-only baseline with prompt `"flooded road"` and no training.
- `ours_no_tm`: FloodRoad-SAM3 with DCA, road prior filtering, LoRA hooks, and CC-RL.
- `ours_tm`: the same model with RG-STM token merging enabled.

The SAM3 wrapper is intentionally isolated in [models/sam3_baseline.py](/Users/ezh/Documents/codes/sam3/floodroad-sam3/models/sam3_baseline.py). Official SAM3 APIs have changed across releases, so the adapter tries common package layouts and fails with a clear integration message if the installed package exposes a different entry point. For smoke tests only, set `sam3.allow_mock: true`; do not use the mock backend for reported results.

## Typical Colab Flow

```bash
cd /content/floodroad-sam3
pip install -r requirements.txt
```

Prepare data from raw SpaceNet 8 files:

```bash
python data/preprocess.py \
  --raw-root /content/spacenet8/raw \
  --processed-root /content/spacenet8/processed \
  --config configs/default.yaml
```

Train DeepLabV3+:

```bash
python train.py --config configs/default.yaml --method deeplab
```

Train FloodRoad-SAM3 without token merging:

```bash
python train.py --config configs/default.yaml --method ours_no_tm
```

Train FloodRoad-SAM3 with RG-STM:

```bash
python train.py --config configs/default.yaml --method ours_tm
```

Evaluate accuracy on the same 20 RL tiles:

```bash
python evaluate.py \
  --config configs/default.yaml \
  --methods deeplab sam_text ours_tm \
  --use-rl-samples \
  --skip-efficiency
```

Run efficiency measurement later on GPU:

```bash
python evaluate.py \
  --config configs/default.yaml \
  --methods deeplab sam_text ours_no_tm ours_tm \
  --efficiency-only
```

## Data Expectations

`preprocess.py` can either discover files by glob patterns from the config or read a pair manifest CSV with these columns:

```text
id,pre_path,post_path,road_geojson_path,flood_path
```

`flood_path` may point to a raster flood mask. If it is absent, preprocessing falls back to flooded-road attributes in the road GeoJSON when available.

Processed tiles are stored as `.npy` arrays and indexed by `manifest.jsonl`:

- pre-disaster RGB tile
- post-disaster RGB tile
- road mask
- flood mask
- flooded-road mask
- integer segment map
- road segment graph JSON

## Important Experiment Note

The config limits the RL fine-tuning set to 20 tiles (`data.rl_sample_limit: 20`) and saves that list in the run directory as `rl_samples.json`. Evaluation can reuse exactly those 20 samples with `--use-rl-samples`, matching the current experiment plan.
