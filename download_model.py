import os
import urllib.request
import sys

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")
            return True
        except Exception as e:
            print(f"Error downloading model: {e}", file=sys.stderr)
            return False
    else:
        print(f"{MODEL_PATH} already exists.")
        return True

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
