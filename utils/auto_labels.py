import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

def rename_images_sequentially(image_folder):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    total_images = len(image_files)
    
    image_files.sort()
    renamed_files = []
    
    for i, filename in enumerate(tqdm(image_files, desc="Renaming images")):
        old_path = os.path.join(image_folder, filename)
        extension = os.path.splitext(filename)[1]
        new_filename = f"temp_{i+1:05d}{extension}"
        new_path = os.path.join(image_folder, new_filename)
        os.rename(old_path, new_path)
        renamed_files.append(new_path)
    
    for i, temp_path in enumerate(tqdm(renamed_files, desc="Finalizing names")):
        final_filename = f"image_{i+1:05d}.jpg"
        final_path = os.path.join(image_folder, final_filename)
        os.rename(temp_path, final_path)
        renamed_files[i] = final_path
    
    print(f"Renamed {total_images} images.")
    return renamed_files

def generate_descriptions(image_folder, renamed_files, cache_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load BLIP model and processor with a specified cache directory
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir).to(device)
    image_descriptions = {}

    for image_file in tqdm(renamed_files, desc="Generating descriptions"):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        
        generated_ids = model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1
        )
        
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        image_descriptions[os.path.basename(image_file)] = description

    return image_descriptions

def save_descriptions_to_json(descriptions, output_json):
    with open(output_json, 'w') as json_file:
        json.dump(descriptions, json_file, indent=4)
    print(f"Descriptions saved to {output_json}")

def main():
    image_folder = "E:/Stable Diffusion rebuild/datasets/images"
    output_json = "./datasets/captions.json"
    cache_dir = "E:/Stable Diffusion rebuild/saved_models"

    os.makedirs(cache_dir, exist_ok=True)

    renamed_files = rename_images_sequentially(image_folder)
    descriptions = generate_descriptions(image_folder, renamed_files, cache_dir)
    save_descriptions_to_json(descriptions, output_json)

if __name__ == "__main__":
    main()