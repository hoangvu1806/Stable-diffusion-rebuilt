import os
from PIL import Image
from transformers import CLIPTokenizer

# Đường dẫn đến dữ liệu
data_dir = "E:\Stable Diffusion rebuild\datasets"
image_dir = os.path.join(data_dir, "images")
captions_file = os.path.join(data_dir, "captions.txt")

# Tải tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Đọc và chuẩn bị dữ liệu
def load_data(image_dir, captions_file):
    with open(captions_file, 'r') as f:
        captions = f.readlines()
    
    image_paths = [os.path.join(image_dir, f"{i}.jpg") for i in range(len(captions))]
    tokenized_captions = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
    
    return image_paths, tokenized_captions

def resize_images(image_dir, size=(64, 64)):
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            with Image.open(img_path) as img:
                img = img.resize(size)
                img.save(img_path)

resize_images("E:\Stable Diffusion rebuild\datasets\images")

image_paths, tokenized_captions = load_data(image_dir, captions_file)
