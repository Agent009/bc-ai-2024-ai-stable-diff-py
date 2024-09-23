import torch
from diffusers import DiffusionPipeline

from lib.utils import get_model, get_generations_path


def generate(prompt="Describe the image you want to generate: ", model_id=get_model(), seed=123456789, n=1):
    print("\n=======================================")
    print("Let's create an AI-powered image via Stable Diffusion!")
    print("=======================================\n")
    # Load the Stable Diffusion model
    pipeline = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    # Set the random/fixed seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed)
    prompt = input(prompt)
    # Generate the images
    images = pipeline(prompt, generator=generator, num_images_per_prompt=n).images
    idx = 0

    for image in images:
        print("Generated image:")
        image.show()
        print("\n")
        # Save the image to disk
        image.save(get_generations_path() + "output_" + str(idx) + ".png")
        idx += 1
