import os
from PIL import Image

# Đường dẫn đến dữ liệu
data_dir = r"E:\Stable Diffusion rebuild\datasets\images_resized"

def resize_images(image_dir, size=(512, 512)):
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist.")
        return

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.resize(size)
                    img.save(img_path)
                    print(f"Resized and saved {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# resize_images(data_dir)

def crop_images(image_dir, target_size=(512, 512)):
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist.")
        return

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                    # Crop ảnh thành hình vuông theo yêu cầu
                    if width > height:
                        left = (width - height) / 2
                        right = left + height
                        top = 0
                        bottom = height
                    else:
                        top = 0
                        bottom = width
                        left = 0
                        right = width
                        if height > width:
                            bottom = top + width
                        
                    img = img.crop((left, top, right, bottom))

                    # Resize ảnh thành 512x512
                    img = img.resize(target_size)
                    img.save(img_path)
                    print(f"Cropped to square and resized {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


crop_images(data_dir)