from lib.imageAI import generate as generate_images
from lib.local import local_txt2img as local_txt2img

# ---------- Main script
if __name__ == '__main__':
    # ---------- Generate images
    # generate_images(n=2)

    # ---------- Local text-to-image generation
    local_txt2img(n=4)
