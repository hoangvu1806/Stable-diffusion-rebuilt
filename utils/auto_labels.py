import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

def rename_images_sequentially(image_folder):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    total_images = len(image_files)
    
    image_files.sort()
    existing_files = [f for f in image_files if f.startswith("imae_")]
    unformal_filenames = [item for item in image_files if item not in existing_files]
    print(f"Renaming {len(unformal_filenames)}/{total_images} unformal images.")

    if existing_files:
        max_existing_num = max(int(f.split('_')[1].split('.')[0]) for f in existing_files)
    else:
        max_existing_num = 0

    renamed_files = []
    max_existing_num = 0

    for i, filename in enumerate(tqdm(unformal_filenames, desc="Renaming images")):
        old_path = os.path.join(image_folder, filename)
        extension = os.path.splitext(filename)[1]
        new_filename = f"tep_{max_existing_num + 1:05d}{extension}"
        new_path = os.path.join(image_folder, new_filename)
        
        if not os.path.exists(new_path):  # Check if the new path exists to avoid collision
            os.rename(old_path, new_path)
            renamed_files.append(new_path)
            max_existing_num += 1
        else:
            print(f"File {new_path} already exists, skipping rename for {old_path}")
    
    new_filenames = []
    max_existing_num = 0

    for i, temp_path in enumerate(tqdm(renamed_files, desc="Finalizing names")):
        final_filename = f"image_{max_existing_num + 1:05d}.jpg"
        final_path = os.path.join(image_folder, final_filename)
        os.rename(temp_path, final_path)
        new_filenames.append(final_path)
        renamed_files[renamed_files.index(temp_path)] = final_path
        max_existing_num += 1
    
    print(f"Renamed {len(renamed_files)}/{total_images} images.")
    return new_filenames

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
            max_length=1000,
            temperature=0.5,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.2,
            num_return_sequences=1
        )
        
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        image_descriptions[os.path.basename(image_file)] = description

    return image_descriptions

def save_descriptions_to_json(descriptions, output_json):
    if os.path.exists(output_json) and os.path.getsize(output_json) > 0:  # Check if the file exists and is not empty
        with open(output_json, 'r') as json_file:
            try:
                existing_descriptions = json.load(json_file)
            except json.JSONDecodeError:
                existing_descriptions = {}  # Handle case where JSON is invalid or empty
    else:
        existing_descriptions = {}

    # Update existing descriptions with new descriptions
    existing_descriptions.update(descriptions)
    
    with open(output_json, 'w') as json_file:
        json.dump(existing_descriptions, json_file, indent=4)
    
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