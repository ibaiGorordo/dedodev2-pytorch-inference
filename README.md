# DeDoDe v2 Pytorch Inference
Repository for performing feature matching using the DeDoDev2 model in PyTorch

![matches](https://github.com/user-attachments/assets/eef82d14-04f9-44a2-96e6-800bcad3f630)

## Installation

### Option1: pip installation
```bash
git clone https://github.com/ibaiGorordo/dedodev2-pytorch-inference.git
cd dedodev2-pytorch-inference
pip install -r requirements.txt
```

### Option2: uv installation
Recommended but requires uv to be installed (https://docs.astral.sh/uv/getting-started/installation/)
```bash
git clone https://github.com/ibaiGorordo/dedodev2-pytorch-inference.git
cd dedodev2-pytorch-inference
uv sync
```

- Additionally, to activate the uv environment: `source .venv/bin/activate` in macOS and Linux or `.venv\Scripts\activate` in Windows

## Examples:

### Image keypoint detection example

```bash
python example_image_keypoints.py
```

### Image pair matching example

```bash
python example_image_pair_matching.py
```

### Webcam feature tracking example

```bash
python example_webcam_feature_tracking.py
```

### Video feature tracking example (https://youtu.be/cDHNvORTRjo?si=aMZt5hiCrqWf3lhQ)

```bash
python example_video_feature_tracking.py
```
![dedodev2](https://github.com/user-attachments/assets/947dc503-6e66-4b51-a54c-1e5e4f27aa43)

## License
The code is taken from the official [DeDoDe repository](https://github.com/Parskatt/DeDoDe) which is distributed under the MIT License.
See [LICENSE](https://github.com/Parskatt/DeDoDe/blob/main/LICENSE) for more information.


### References:
- DeDoDe: https://github.com/Parskatt/DeDoDe
