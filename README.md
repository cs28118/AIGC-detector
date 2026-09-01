# AIGC Detector

## Project Overview

AIGC Detector is an image-classification system that predicts whether an image is real or AI-generated. The project also includes confidence calibration using temperature scaling and robustness evaluation on image data. This helps assess whether the model’s confidence scores remain meaningful when images are transformed.

The detector deliberately fuses semantic and forensic evidence instead of treating AIGC detection as a clean-image classification task.

* **Spatial branch:** an ImageNet-initialised ConvNeXt-Small backbone learns semantic and texture cues.
* **Forensic branch:** a differentiable multi-domain feature extractor operates on image luminance. It combines a high-pass residual and two-scale Haar-wavelet detail maps, three bands of 8x8 block-DCT energy, and a windowed global FFT magnitude spectrum.
* **Robust training:** every training sample yields aligned clean and damaged views. The damaged view receives one or two sampled transforms: JPEG recompression, Gaussian blur, downscale/upscale, Gaussian noise, colour jitter, or centre crop. Classification loss is supplemented by prediction, feature, and forensic-reliability consistency losses.

The manifest builder assigns deterministic group-safe train/validation/calibration splits. Evaluation corruptions are deterministic for each image and seed, enabling fair comparisons. The best checkpoint is selected by an equal-weight clean/robust AUC score, rather than clean AUC alone.

> Selection score = 0.5 x AUC + 0.5 x Accuracy@0.5

## Features

- Detects real and AI-generated images
- Produces a confidence score for each prediction
- Supports batch inference on an image directory
- Uses temperature scaling for confidence calibration
- Evaluates performance on clean and transformed images
- Saves prediction results in JSON format

## Tech stack Used

- Python
- PyTorch
- ConvNeXt Small
- pandas
- scikit-learn
- Google Colab

## Access to our model trained

The pytorch file is available for download under release `v2.0 model`, tag `v2.0`

## Setup and Installation

Prerequisites: Python 3.10+ and PyTorch-compatible hardware. CUDA available GPU is highly recommended. CPU execution is supported.

### Clone the Repository and Download Dependencies

```powershell
git clone https://github.com/cs28118/AIGC-detector.git
cd AIGC-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.

## Running Inference

To classify all images in a directory:

```powershell
python -m src.infer \
  --input-dir input_images_dir \
  --checkpoint model_path \
  --output artifacts/predictions.json \
```

The output JSON file contains the image path and prediction:

```
[
  {
    "image_path": "path/to/images/image1.jpg",
    "pred": 0.9132
  },
  {
    "image_path": "path/to/images/image2.png",
    "pred": 0.0745
  }
]
```

The `pred` value represents the estimated likelihood that the image is AI-generated. Values closer to `1` indicate a higher likelihood of AI generation, while values closer to `0` indicate a higher likelihood that the image is real.

## Reproducing Our Results

1. Download the Dataset

The project uses the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) from Kaggle.

```powershell
kaggle datasets download \
  -d birdy654/cifake-real-and-ai-generated-synthetic-images \
  -p data/cifake \
  --unzip
```

> We also use [GANGen](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) and [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) datasets. For these two dataset, we pick similar amount from each subfolder and combine it with CiFake dataset.

```
Sampling cifake_colab.csv
Available per label: {'train': 41757, 'val': 4967, 'calibration': 2356}
Selected per label: {'train': 2550, 'val': 300, 'calibration': 150}

Sampling universalfake_1.csv
Available per label: {'train': 852, 'val': 100, 'calibration': 48}
Selected per label: {'train': 852, 'val': 100, 'calibration': 48}

Sampling universalfake_2.csv
Available per label: {'train': 853, 'val': 100, 'calibration': 47}
Selected per label: {'train': 853, 'val': 100, 'calibration': 47}

Sampling universalfake_3.csv
Available per label: {'train': 853, 'val': 104, 'calibration': 43}
Selected per label: {'train': 853, 'val': 104, 'calibration': 43}

Sampling universalfake_4.csv
Available per label: {'train': 848, 'val': 99, 'calibration': 53}
Selected per label: {'train': 848, 'val': 99, 'calibration': 53}

Sampling universalfake_5.csv
Available per label: {'train': 841, 'val': 110, 'calibration': 49}
Selected per label: {'train': 841, 'val': 110, 'calibration': 49}

Sampling universalfake_6.csv
Available per label: {'train': 867, 'val': 89, 'calibration': 44}
Selected per label: {'train': 867, 'val': 89, 'calibration': 44}

Sampling universalfake_7.csv
Available per label: {'train': 837, 'val': 110, 'calibration': 53}
Selected per label: {'train': 837, 'val': 110, 'calibration': 53}

Sampling universalfake_8.csv
Available per label: {'train': 843, 'val': 120, 'calibration': 37}
Selected per label: {'train': 843, 'val': 120, 'calibration': 37}

Sampling self_synthesis_1.csv
Available per label: {'train': 1689, 'val': 190, 'calibration': 93}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Sampling self_synthesis_2.csv
Available per label: {'train': 1686, 'val': 209, 'calibration': 104}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Sampling self_synthesis_3.csv
Available per label: {'train': 1662, 'val': 207, 'calibration': 102}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Sampling self_synthesis_4.csv
Available per label: {'train': 1706, 'val': 185, 'calibration': 90}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Sampling self_synthesis_5.csv
Available per label: {'train': 1689, 'val': 195, 'calibration': 104}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Sampling self_synthesis_6.csv
Available per label: {'train': 1706, 'val': 207, 'calibration': 84}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Sampling self_synthesis_7.csv
Available per label: {'train': 1682, 'val': 188, 'calibration': 105}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Sampling self_synthesis_8.csv
Available per label: {'train': 1688, 'val': 206, 'calibration': 94}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Sampling self_synthesis_9.csv
Available per label: {'train': 1678, 'val': 207, 'calibration': 104}
Selected per label: {'train': 850, 'val': 100, 'calibration': 50}

Final split and label counts:
split        label
calibration  0          974
             1          974
train        0        16994
             1        16994
val          0         2032
             1         2032
```

2. Build an auditable manifest. Defaults create stable train (85%), validation (10%), and calibration (5%) splits, remove exact duplicates, and write an exclusion audit.

```powershell
python scripts/build_manifest.py \
  --real-root data/cifake/train/REAL \
  --ai-root data/cifake/train/FAKE \
  --output manifests/cifake.csv \
  --source cifake \
  --ai-generator stable_diffusion \
  --val-fraction 0.10 \
  --calibration-fraction 0.05 \
  --seed 42
```

3. Train the redistribution-aware detector. Selection evaluates clean, JPEG quality 30, blur sigma 2.0, 0.25x resize/upscale, and 80% centre crop. `best.pt` is selected by `0.5 x clean AUC + 0.5 x mean corrupted AUC`; `last.pt` is resumable.

```powershell
python -m src.train \
  --manifest manifests/cifake.csv \
  --output-dir artifacts/convnext_small_robust \
  --architecture convnext_small \
  --epochs 3 \
  --batch-size 16 \
  --workers 2 \
  --selection-workers 0 \
  --selection-max-samples 2000 \
  --learning-rate 2e-4 \
  --device cuda
```

For CPU training, replace `--device cuda` with `--device cpu`.

4. Calibrate probabilities on the separate calibration split. Temperature scaling changes confidence calibration without changing ranking/AUC.

```powershell
python -m src.calibrate \
  --checkpoint artifacts/convnext_small_robust/best.pt \
  --manifest manifests/cifake.csv \
  --split calibration \
  --output artifacts/convnext_small_robust/temperature.json \
  --batch-size 64 \
  --workers 2 \
  --device cuda
```

5. Evaluate all documented redistribution conditions. The CSV includes ROC-AUC, threshold-0.5 accuracy, mean robust AUC, and the combined score.

```powershell
python -m src.evaluate \
  --checkpoint artifacts/convnext_small_robust/best.pt \
  --manifest manifests/cifake.csv \
  --split val \
  --calibration artifacts/convnext_small_robust/temperature.json \
  --output artifacts/convnext_small_robust/robustness.csv \
  --batch-size 64 \
  --workers 2 \
  --device cuda
```

> The evaluation results will be saved to:
> artifacts/convnext_small_robust/robustness.csv

## Limitations

The model is trained primarily on the limited dataset, so its performance may not generalize to every type of real or AI-generated image. New image-generation models may produce visual patterns that differ from those seen during training. 

The system returns a probability, not provenance therefore should be treated as an estimate rather than definitive proof that an image is AI-generated. The confidence score also depends on the quality and diversity of the training data. Even after calibration, the model may be uncertain or overconfident when given unfamiliar images.

The prototype also lacks adversarial-attack evaluation, metadata provenance checks, image-region localisation, and production serving/monitoring.

## Future Improvements

Given more time, we would:

1. Train with a more diverse dataset.
2. Include a wider variety of real-world photographs
3. Test additional transformations and compression levels
4. Improve cross-dataset and cross-generator evaluation
5. Perform more detailed false-positive and false-negative analysis
6. Build a web UI for easy access to model by image upload and result visualization
7. Explore more ensemble models and vision transformers

## Robustness Evaluation Summary

The following is the result of applying different transformation to test dataset and eval using the latest 'v2.0_model.py`.

| Condition | ROC AUC | Accuracy@0.5 | N |
| :--- | :--- | :--- | :--- |
| clean | 0.997094 | 0.980069 | 4064 |
| jpeg30 | 0.966645 | 0.897884 | 4064 |
| jpeg50 | 0.977111 | 0.926181 | 4064 |
| jpeg70 | 0.981740 | 0.925689 | 4064 |
| jpeg90 | 0.990953 | 0.949803 | 4064 |
| blur0.5 | 0.999351 | 0.988435 | 4064 |
| blur1.0 | 0.974512 | 0.938238 | 4064 |
| blur2.0 | 0.965614 | 0.878691 | 4064 |
| resize0.5 | 0.986163 | 0.956201 | 4064 |
| resize0.25 | 0.953449 | 0.887303 | 4064 |
| noise0.02 | 0.992341 | 0.961614 | 4064 |
| noise0.05 | 0.981170 | 0.929134 | 4064 |
| noise0.10 | 0.941608 | 0.860974 | 4064 |
| color | 0.995373 | 0.973179 | 4064 |
| crop | 0.996847 | 0.981545 | 4064 |
| mean_robust | 0.978777 | NaN | 4064 |
| combined_score | 0.987935 | NaN | 4064 |

The equal-weight clean/robust AUC is **0.9971**. JPEG, strong blur, strong noise, massive resize (0.25) highly affect the result.

## Project tools and reference

* **Language and framework:**
  - Python, PyTorch, TorchVision (ConvNeXt-Small), NumPy, Pillow, pandas, scikit-learn, tqdm
  - Exact minimum versions are listed in `requirements.txt`.
* **Project scripts:**
  - `scripts/build_manifest.py` creates deduplicated group-safe manifests
  - `src/train.py`, `src/calibrate.py`, `src/evaluate.py`, and `src/infer.py` cover the model lifecycle.
* **Datasets:**
  1. [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
  2. [GANGen](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)
  3. [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
