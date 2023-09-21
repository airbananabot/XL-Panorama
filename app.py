from potassium import Potassium, Request, Response
from diffusers import StableDiffusionXLPipeline
# from diffusers import DiffusionPipeline
import torch
import base64
import requests
import os
from io import BytesIO

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA = 'https://civitai.com/api/download/models/143197'

app = Potassium("stable-diffusion-xl-base-1.0")


import requests

def download_from_url(url: str, destination_path: str) -> bool:
    try:
        response = requests.get(url, allow_redirects=True, headers={"content-disposition": "attachment"})
        if response.status_code == 200:
            with open(destination_path, 'wb') as file:
                file.write(response.content)
            return True
        else:
            print(f"Failed to download from {url}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading from {url}. Error: {e}")
        return False

# Example usage




@app.init
def init():
    """Initialize the application with the model."""
    print("Initializing...")
    # Initialize the model from StableDiffusionXLPipeline
    model = StableDiffusionXLPipeline.from_pretrained(MODEL, use_safetensors=True).to("cuda")
    # model = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    print("Model initialized...")
    # refiner = DiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-refiner-1.0",
    #     text_encoder_2=model.text_encoder_2,
    #     vae=model.vae,
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     variant="fp16",
    # ).to("cuda")
    # url = 'https://civitai.com/api/download/models/143197'
    # lora_path = 'ckpt/360XL.safetensors'
    # lora_path = '360XL.safetensors'
    # lora_ready = False
    # if(os.path.exists(lora_path)):
    #     print("Lora already downloaded!")
    #     lora_ready = True
    # else:
    #     print("Downloading Lora...")
    #     success = download_from_url(LORA, lora_path)
    #     if success:
    #         print("Download successful!")
    #         lora_ready = True
    #     else:
    #         print("Lora Download failed!")
    # if not lora_ready:
    #     print("Lora not ready!")
    # else:
    #     print("Loading Lora...")
    #     try:
    #         model.load_lora_weights(lora_path)
    #         print("Lora loaded!")
    #     except Exception as e:
    #         print(f"Error loading lora from {lora_path}. Error: {e}")
            
    context = {
        "model": model
    }
    return context

@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate image from a prompt."""
    model = context.get("model")
    prompt = request.json.get("prompt")
    width = request.json.get("width")
    height = request.json.get("height")
    
    # Assuming the pipeline method remains the same
    images = model(prompt=prompt,width=width,height=height).images[0]
    buffered = BytesIO()
    images.save(buffered, format="JPEG", quality=80)
    img_str = base64.b64encode(buffered.getvalue())
    
    return Response(json={"output": str(img_str, "utf-8")}, status=200)

if __name__ == "__main__":
    app.serve()
