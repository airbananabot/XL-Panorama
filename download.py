from diffusers import DiffusionPipeline
# from diffusers import StableDiffusionXLPipeline
import torch
import requests

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA = 'https://civitai.com/api/download/models/143197'

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

def download_model() -> tuple:
    """Download the model"""
    # model = StableDiffusionXLPipeline.from_pretrained(MODEL, use_safetensors=True)
    model = DiffusionPipeline.from_pretrained(MODEL, use_safetensors=True)

    # lora_path = 'ckpt/360XL.safetensors'
    lora_path = '360XL.safetensors'
    success = download_from_url(LORA, lora_path)
    if success:
        print("Download successful!")
        try:
            model.load_lora_weights(lora_path)
            print("Lora loaded!")
        except Exception as e:
            print(f"Error loading lora from {lora_path}. Error: {e}")
    else:
        print("Lora Download failed!")
    print("download_model() complete!")

if __name__ == "__main__":
    download_model()
    