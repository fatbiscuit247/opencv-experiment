# opencv-experiment

Experimenting with opencv's filters. Currently using webcam

A real-time beauty filter using OpenCV's DNN face detector and bilateral smoothing.  
Built as a side project to explore computer vision fundamentals — face detection, facial masking, and selective image processing.

## How it works

1. **Face detection** — OpenCV's res10 SSD DNN model locates faces in each frame
2. **Feathered mask** — an elliptical mask is built around each detected face with soft edges to avoid hard compositing lines
3. **Bilateral smoothing** — applied only to the face region; preserves edges (eyes, lips, brows) while smoothing skin texture
4. **Skin tone correction** — subtle warmth boost and redness reduction
5. **Alpha compositing** — the smoothed face region is blended back onto the original frame

This approach is meaningfully different from naive circular blurring — it targets skin specifically and preserves facial features.

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/opencv-experiment.git
cd opencv-beautyfilter

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model files
python download_models.py

# 4. Run
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save snapshot as `snapshot.jpg` |
| `r` | Edit in `filter.py` |

## Project structure

```
opencv-beautyfilter/
├── main.py             # Webcam loop and window management
├── filter.py           # Beauty filter pipeline
├── download_models.py  # One-time model downloader
├── requirements.txt
└── models/             # DNN model files (downloaded, not committed)
```

## Next steps

- [ ] Facial landmark detection (68-point) for feature-aware masking
- [ ] Frequency separation for texture-preserving smoothing  
- [ ] Skin segmentation to exclude non-skin areas from processing
- [ ] Adapter to swap webcam input for DSLR frames (for photobooth integration)
