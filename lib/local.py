import base64
import os

import requests
from dotenv import load_dotenv, find_dotenv

from lib.utils import get_generations_path

# ---------- Load environment variables
load_dotenv(find_dotenv())
LOCAL_API = os.getenv("LOCAL_API")


# ---------- Entrypoint point for local API calls
def run_local():
    local_txt2img()

# ---------- Setup
def get_endpoint(mode="txt2img" or "img2img"):
    return "txt2img" if mode == "txt2img" else "img2img"


def get_url(mode="txt2img"):
    return f"{LOCAL_API}{get_endpoint(mode)}"

# ---------- Generations
def local_txt2img(prompt="What would you like to generate? \n",
                  negative_prompt="What would you like to avoid in the image? \n",
                  input_prompt="How many steps would you like to take? \n",
                  n=2):
    payload = {
        "prompt": input(prompt),
        "negative_prompt": input(negative_prompt),
        "steps": int(input(input_prompt)),
        "n_iter": n
    }
    response = requests.post(url=get_url(mode="txt2img"), json=payload).json()
    handle_generated_images(response['images'])


def handle_generated_images(images, prefix="local_generated_"):
    idx = 0

    for image in images:
        try:
            with open(get_generations_path() + f"{prefix}{idx}.png", 'wb') as f:
                f.write(base64.b64decode(image))
        except Exception as e:
            print(f"Error saving image: {e}")

        idx += 1
