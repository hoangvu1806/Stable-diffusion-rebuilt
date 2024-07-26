
# Stable Diffusion Project

This project aims to build a Stable Diffusion model using PyTorch. The model consists of a VAE (Variational Autoencoder) and a U-Net architecture for generating images.

## Project Structure
### stable_diffusion_rebuilt_project/
- │
- ├── data/
- │ ├── raw/ # Thư mục chứa dữ liệu thô
- │ ├── processed/ # Thư mục chứa dữ liệu đã được tiền xử lý
- │
- ├── models/
- │ ├── vae.py # File định nghĩa mô hình VAE
- │ ├── unet.py # File định nghĩa mô hình U-Net
- │ ├── stable_diffusion.py # File định nghĩa mô hình Stable Diffusion kết hợp VAE và U-Net
- │
- ├── scripts/
- │ ├── train.py # File script huấn luyện mô hình
- │ ├── generate.py # File script tạo sinh hình ảnh
- │
- ├── utils/
- │ ├── dataset.py # File định nghĩa và xử lý dataset
- │ ├── preprocessing.py # File chứa các hàm tiền xử lý dữ liệu
- │
- ├── saved_models/ # Thư mục chứa các mô hình đã được huấn luyện và lưu lại
- │
- ├── env/ # Thư mục chứa cấu hình môi trường ảo
- │ ├── environment.yml # File cấu hình môi trường ảo (nếu dùng conda)
- │ ├── requirements.txt # File liệt kê các thư viện cần thiết
- │
- ├── Dockerfile # Dockerfile để xây dựng Docker image
- ├── README.md # File README mô tả dự án
- ├── .gitignore # File .gitignore để bỏ qua các file không cần thiết khi sử dụng Git

```bash
conda create --prefix ./env python=3.8
conda activate ./env
conda install -c conda-forge diffusers transformers safetensors