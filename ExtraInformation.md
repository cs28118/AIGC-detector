## 1. What Makes an AI Image Detectable?

Real cameras and generative models leave different fingerprints — detectors learn to spot them:

* **Frequency artifacts**: GAN/diffusion up-sampling leaves periodic patterns in the Fourier spectrum that cameras don't produce.
* **Noise & sensor fingerprints**: real photos carry sensor noise (PRNU); synthetic images lack it or fake it imperfectly.
* **Texture & fine detail**: skin, hair, foliage, text and reflections are where models still slip.
* **Semantic / physics tells**: impossible lighting, warped hands, garbled text, inconsistent shadows.

### Key Insight: Go hybrid

* Best detectors combine **high-level CLIP semantics** + **low-level frequency patches** — each catches what the other misses, and both survive different transforms.
* Many signals live in **high-frequency detail** — exactly what compression and blur destroy. That's why robustness is hard.
* **Don't just fine-tune a classifier.** Think about what your model is actually learning — is it a real artifact, or a dataset shortcut?

## 2. A Baseline Detection Pipeline

Start simple. A clean binary classifier you can actually finish beats a fancy model you can't.

[Input Image] ➔ [Preprocess (resize / normalise)] ➔ [Backbone (CNN / ViT)] ➔ [Classifier Head] ➔ [Real vs AI + confidence]

* Fine-tune a pretrained backbone (ResNet / EfficientNet / ViT) as a binary classifier — a strong day-1 baseline.
* **Optional upgrade**: add a frequency branch (FFT / DCT features) and fuse it with the spatial branch.
* Output a **calibrated probability**, not just a label — you'll need it for thresholding & error analysis.

## 3. The Key Idea: Train for the Real World

The biggest lever isn't a fancier model — it's **what you train on**.

[Clean Images] ➔ [Random Transforms (blur · JPEG · crop · colour · noise)] ➔ [Augmented Views] ➔ [Model learns signals that survive]

* **Augmentation = simulate redistribution during training**: JPEG-compress, blur, resize, crop, colour-jitter, add noise, re-screenshot.
* **SAFE insight (KDD 2025)**: crop instead of down-sample to preserve high-freq artifacts; ColorJitter + RandomRotation kill colour/semantic shortcuts.
* **DDA insight (NeurIPS 2025)**: watch out for frequency bias — JPEG in your real images can become a spurious signal. Align pixel + frequency.
* **Augmentation + data alignment > architecture tricks**. Training-pipeline improvements beat fancier backbones.

> **Mental model**: *"If a transformation can happen on a real feed, it must happen in your training pipeline."*


## 4. Evaluate Like It's the Real World

Clean accuracy is the #1 way to fool yourself.

* Build a **transformed test set**: e.g., JPEG q=90/70/50/30.
* **Primary metric**: ROC AUC (threshold-free, robust to imbalance).
* **Final Score** = 0.50×AUC_clean + 0.50×AUC_robust
* **Cross-generator**: test on generators NOT in training — the real generalization test.

| Condition | Acc. | AUC |
| :--- | :--- | :--- |
| Clean | 0.97 | 0.99 |
| JPEG q30 | 0.86 | 0.93 |
| Blur σ=2 | 0.82 | 0.90 |
| Crop 80% | 0.88 | 0.94 |
| Unseen gen. | 0.71 | 0.80 |

**Judges want a compact robustness table + error-analysis note.**

## Competition Rules — Know Before You Build

* **Open-source only:** all pretrained backbones must be public (ResNet, ViT, CLIP, DINOv2, etc.). Custom architectures must be released under MIT/Apache.
* **Winning teams open-source:** training pipelines, hyperparameters, evaluation code, and model weights.
* **Data:** only public/licensed datasets (e.g., WildFake, CIFAKE, SID_Set). No proprietary/production data, no test-label training.
* **Augmentation:** you may generate transformed samples from approved datasets; include generation scripts for reproducibility.
* **Model limits:** **You may use models with < 2B parameters! Do not directly replicate existing models or approaches.**
* **Submission:** public GitHub repo + run script + Devpost description + 2–4 min YouTube demo.

*Violations result in disqualification. Organizers reserve final authority on all matters.*