import os
import json
import torch, re
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# Function to rename images sequentially
def rename_images_sequentially(image_folder) -> list:
    i = 1
    renamed_files = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            dst = "image_" + str(i) + ".jpg"
            src = os.path.join(image_folder, filename)
            dst = os.path.join(image_folder, dst)
            
            # Kiểm tra xem tên tệp mới đã tồn tại chưa
            while os.path.exists(dst):
                i += 1
                dst = "image_" + str(i) + ".jpg"
                dst = os.path.join(image_folder, dst)
            
            # Đổi tên tệp
            os.rename(src, dst)
            renamed_files.append(dst)
            i += 1
            
    return renamed_files

# Function to generate descriptions using BLIP with adjustable parameters
def generate_descriptions(image_folder, renamed_files, cache_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using device: {device}")
    
    # Load BLIP model and processor with a specified cache directory
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir).to(device)

    image_descriptions = {}

    for image_file in tqdm(renamed_files):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Adjusting parameters for generation
        outputs = model.generate(
            **inputs,
            max_length=500,         # Increase for longer descriptions
            temperature=0.7,        # Adjust for more diverse or focused descriptions
            top_k=50,               # Use top-k sampling
            top_p=0.9,              # Use nucleus sampling
            repetition_penalty=1.2  # Penalize repetition
        )
        
        description = processor.decode(outputs[0], skip_special_tokens=True)
        image_descriptions[os.path.basename(image_file)] = description

    return image_descriptions

# Function to save descriptions to a JSON file
def save_descriptions_to_json(descriptions, output_json):
    with open(output_json, 'w') as json_file:
        json.dump(descriptions, json_file, indent=4)
    print(f"Descriptions saved to {output_json}")

# Define the input folder, output JSON file, and cache directory
image_folder = "E:/Stable Diffusion rebuild/datasets/images"
output_json = "./datasets/captions.json"
cache_dir = "E:\\Stable Diffusion rebuild\\saved_models"  # Directory where the model will be cached

# Ensure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

# Rename images sequentially
renamed_files = rename_images_sequentially(image_folder)

# Generate descriptions for renamed images
descriptions = generate_descriptions(image_folder, renamed_files, cache_dir)

# Save descriptions to JSON file
save_descriptions_to_json(descriptions, output_json)
