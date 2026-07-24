# 🌿 Plant Disease Classifier — Streamlit App

A Streamlit web app that serves the CNN trained in `plant_disease_prediction.ipynb`
(PlantVillage dataset, Keras/TensorFlow) for leaf disease prediction from an
uploaded image. Includes a Dockerfile so you can run it anywhere.

```
plant-disease-app/
├── app.py                       # Streamlit app
├── class_indices.json           # index -> class name mapping
├── regenerate_class_indices.py  # how to regenerate the mapping if needed
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── models/
│   └── (put plant_disease_model.h5 here — not included, see below)
└── README.md
```

## 0. Get the model file — no manual download needed

This zip does **not** include `plant_disease_model.h5` — it's a trained
model file, too large to bundle or push to GitHub normally. Your notebook
already saves it to Google Drive here:

```python
model_save_path = '/content/drive/MyDrive/plant_disease_model.h5'
model.save(model_save_path)
```

Instead of downloading that file to your laptop and re-uploading it
somewhere, `app.py` will **fetch it straight from Google Drive
automatically** the first time it starts, using
[`gdown`](https://pypi.org/project/gdown/) (already in `requirements.txt`).
You only need to point it at the right file, once:

1. In Google Drive, right-click `plant_disease_model.h5` → **Share** → change
   access to **"Anyone with the link"** (Viewer is enough).
2. Copy the share link. It looks like:
   `https://drive.google.com/file/d/1AbCDeFGhIJkLmNoPQRstuVWxyz/view?usp=sharing`
   The long string between `/d/` and `/view` is the **file ID** —
   here that's `1AbCDeFGhIJkLmNoPQRstuVWxyz`.
3. Give the app that ID as `GDRIVE_FILE_ID`, using whichever matches where
   you're running it:
   - **Locally**: copy `.streamlit/secrets.toml.example` to
     `.streamlit/secrets.toml` and paste the ID in
     (`.streamlit/secrets.toml` is gitignored — it's never pushed), or
     just run `GDRIVE_FILE_ID=1AbCDe... streamlit run app.py`.
   - **Streamlit Community Cloud**: app settings → **Secrets** → paste
     `GDRIVE_FILE_ID = "1AbCDe..."`.
   - **Docker**: `docker run -p 8501:8501 -e GDRIVE_FILE_ID=1AbCDe... plant-disease-app`.

That's it — no `.h5` file ever touches your local disk or your git repo.
The app downloads it into `models/plant_disease_model.h5` on first launch
and reuses it after that (cached via `@st.cache_resource`, and persisted on
disk between restarts as long as the container/filesystem isn't wiped).

**If you'd rather do it manually instead** (e.g. no internet access from
the deployment target), you can still skip Drive entirely:
- Put the file in this project at `models/plant_disease_model.h5`, **or**
- Point at wherever you already saved it with `MODEL_PATH`:
  ```bash
  MODEL_PATH=/full/path/to/plant_disease_model.h5 streamlit run app.py
  ```
  Docker equivalent:
  ```bash
  docker run -p 8501:8501 \
    -v /full/path/to/your/models:/models \
    -e MODEL_PATH=/models/plant_disease_model.h5 \
    plant-disease-app
  ```

**Important — verify class_indices.json**: `class_indices.json` in this repo
is the standard 38-class PlantVillage ordering (alphabetical, from the
`color` folder). This is very likely correct if you trained on the
unmodified dataset like the notebook does, but folder-listing order can in
theory differ by OS/filesystem. To be 100% sure, see
`regenerate_class_indices.py` for a snippet to run in the notebook that
regenerates this file from your actual `train_generator`.

## 1. Run locally in VS Code (no Docker)

```bash
# from inside plant-disease-app/
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## 2. Run with Docker

Build the image:

```bash
docker build -t plant-disease-app .
```

Run it:

```bash
docker run -p 8501:8501 plant-disease-app
```

Open http://localhost:8501.

If you'd rather keep the model file outside the image and mount it at
runtime (keeps the image small, easier to rebuild):

```bash
# don't COPY the model in the Dockerfile — instead:
docker run -p 8501:8501 -v $(pwd)/models:/app/models plant-disease-app
```

## 3. Push to GitHub

Since the model now stays on Google Drive and is downloaded automatically
(section 0), the repo you push is just code — small and fast, no Git LFS
needed. `.gitignore` already excludes `models/*.h5` and
`.streamlit/secrets.toml` so you won't accidentally commit either.

**Option A — from your own machine**, after unzipping this project:
```bash
git init
git add .
git commit -m "Plant disease classifier app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

**Option B — straight from the Colab notebook, no local download at all.**
If you'd rather not download anything to your laptop, run this in a Colab
cell right after `model.save(...)`. It clones your (empty) GitHub repo,
copies these project files into it, and pushes — all inside Colab:

```python
# One-time: create an empty repo on github.com first, then a token at
# https://github.com/settings/tokens (classic, "repo" scope).
GITHUB_USER = "your-username"
GITHUB_TOKEN = "ghp_xxx..."          # better: store in Colab "Secrets" (key icon)
GITHUB_REPO = "plant-disease-app"

!git clone https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git
# Copy this unzipped project's files into {GITHUB_REPO}/ here (e.g. via
# Google Drive if you uploaded the zip there, or recreate app.py etc. with
# %%writefile). Do NOT copy the .h5 file in — it's meant to stay on Drive.
%cd {GITHUB_REPO}
!git add .
!git commit -m "Plant disease classifier app"
!git push
```

Either way, the model file itself is never part of the push — only its
Google Drive file ID (set as a secret/env var per section 0) travels with
the deployment.

## 4. Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (step 3).
2. Go to https://share.streamlit.io, sign in, click "New app".
3. Pick your repo, branch `main`, main file path `app.py`.
4. Click **"Advanced settings" → Secrets** and paste:
   ```
   GDRIVE_FILE_ID = "1AbCDeFGhIJkLmNoPQRstuVWxyz"
   ```
   (your actual file ID from section 0).
5. Deploy. Streamlit Cloud installs from `requirements.txt` (including
   `gdown`) automatically, and `app.py` pulls the model from Drive on first
   boot — no manual upload step anywhere.

## Notes

- The model expects 256×256 RGB input, matching how it was trained
  (`image_size = (256, 256)` in the notebook).
- `tensorflow-cpu` is used in `requirements.txt` to keep the Docker image
  smaller and because Streamlit Cloud / most basic hosting has no GPU. Swap
  to `tensorflow` if you're deploying somewhere with GPU support you want
  to use.
