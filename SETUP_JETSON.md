# Jetson Nano Setup (Inference)

Step-by-step setup for running this project's **inference** path on an original
**NVIDIA Jetson Nano 4GB**.

**Target platform (do not deviate):**

| Component  | Version                          |
| ---------- | -------------------------------- |
| JetPack    | 4.6 (L4T R32.6.x)                |
| Python     | 3.6.9 (system)                   |
| CUDA       | 10.2 (`/usr/local/cuda-10.2`)    |
| TensorRT   | 8.0.x (system, `python3-libnvinfer`) |
| OpenCV     | 4.1.1 (system, with GStreamer)   |

> All commands run on the Jetson over SSH or a local terminal. They assume the
> default `nano` user. Adjust paths if you cloned the repo elsewhere; below it is
> assumed at `~/Project-MedicalSeg`.

---

## 1. Add a 4 GB swap file

The Nano has only 4 GB of RAM. Building `pycuda` and loading a TensorRT engine
can exhaust it, so add swap first. (Skip if you already run `nvzram`/zram and a
swap file — but a real file is more reliable for large builds.)

```bash
# Create a 4 GB swap file
sudo fallocate -l 4G /var/swapfile
# If fallocate is unavailable on your filesystem, use dd instead:
# sudo dd if=/dev/zero of=/var/swapfile bs=1M count=4096

sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile

# Make it persistent across reboots
echo '/var/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h          # "Swap" total should now show ~4.0Gi
swapon --show
```

---

## 2. Create a virtualenv with `--system-site-packages`

This is the critical flag. `tensorrt` and `cv2` are installed as **system**
packages by JetPack into `/usr/lib/python3.6/dist-packages` and are **not** on
PyPI for Jetson. A normal isolated venv would hide them. `--system-site-packages`
lets the venv import the system `tensorrt`/`cv2` while still letting us pip-install
our own pinned packages on top.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev

cd ~/Project-MedicalSeg
python3 -m venv --system-site-packages .venv-jetson
source .venv-jetson/bin/activate

# Upgrade just pip; keep setuptools/wheel modest for Python 3.6 compatibility
python -m pip install --upgrade "pip<21.4"
```

Confirm the system packages are visible from inside the venv:

```bash
python -c "import cv2; print('cv2', cv2.__version__)"
python -c "import tensorrt as trt; print('tensorrt', trt.__version__)"
```

Both should print versions (cv2 ~4.1.1, tensorrt ~8.0). If `tensorrt` is missing,
install the system package and recreate the venv:

```bash
sudo apt-get install -y python3-libnvinfer python3-libnvinfer-dev
```

---

## 3. Export CUDA environment variables (before installing pycuda)

`pycuda` compiles against CUDA 10.2 at install time. It needs `nvcc` on `PATH`
and the CUDA runtime libraries on `LD_LIBRARY_PATH`, otherwise the build fails
with "Could not find nvcc" or a link error.

```bash
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda-10.2

# Verify nvcc is now found and reports release 10.2
nvcc --version
```

To make these permanent, append them to your shell profile:

```bash
cat >> ~/.bashrc <<'EOF'
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda-10.2
EOF
```

---

## 4. Install the Jetson inference requirements

With the venv active and the CUDA env vars exported in the **same** shell:

```bash
cd ~/Project-MedicalSeg
pip install -r requirements-jetson.txt
```

Notes:

- The `pycuda` build can take several minutes and is RAM-hungry — this is why
  the swap file in step 1 matters.
- This file deliberately does **not** install `tensorflow`, `opencv-python`, or
  `Pillow`. `cv2` and `tensorrt` come from the system packages exposed by
  `--system-site-packages`; pip's `opencv-python` lacks GStreamer support and
  must not shadow the system build.

---

## 5. Verify the install

Run the backend selector in backend-only mode. It imports `tensorrt`, `pycuda`,
`cv2`, and `numpy`, then reports which inference backend it would use — without
needing an input image:

```bash
cd ~/Project-MedicalSeg
python jetson_inference.py --backend-only
```

Expected output (once a TensorRT engine exists at `models/segmentation.trt`):

```
[Inference] backend=tensorrt path=models/segmentation.trt
Selected backend: tensorrt
```

If no `.trt` engine is present yet, it falls back and prints a different backend
(`tflite` or `keras`) along with the reason — that still confirms the Python
environment and imports are healthy. To exercise the full TensorRT path, build
`models/segmentation.trt` from the exported ONNX model with `trtexec`, then
re-run the command above.

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `import tensorrt` fails inside venv | Venv was created without `--system-site-packages`, or `python3-libnvinfer` is missing. Reinstall the apt package and recreate the venv. |
| pip tries to build `opencv-python` | Something pulled it in transitively. It is **not** in `requirements-jetson.txt`; do not add it. Use the system `cv2`. |
| `pycuda` build: "Could not find nvcc" | The CUDA env vars from step 3 were not exported in this shell. Re-export and retry. |
| Out-of-memory during `pip install` | Swap not active. Re-check step 1 (`swapon --show`). |
| `cv2` shows a pip version, not 4.1.1 | A pip `opencv-python` is shadowing the system build. `pip uninstall opencv-python` and rely on the system package. |
