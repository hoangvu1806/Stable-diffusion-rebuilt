import torch, os
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import Generator
from discriminator import Discriminator
from tqdm import tqdm

# Thiết lập các tham số
latent_dim = 100
img_shape = (3, 64, 64)
lr = 2e-3
batch_size = 64
n_epochs = 200

# Đường dẫn đến dữ liệu
data_dir = "E:\Stable Diffusion rebuild\datasets"
image_dir = os.path.join(data_dir, "images")
captions_file = os.path.join(data_dir, "captions.txt")

# Định nghĩa hàm mất mát và optimizer
adversarial_loss = torch.nn.BCELoss()
generator = Generator(latent_dim, img_shape).cuda()
discriminator = Discriminator(img_shape).cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Dataset và DataLoader
class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, tokenized_captions, transform=None):
        self.image_paths = image_paths
        self.tokenized_captions = tokenized_captions
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = self.tokenized_captions["input_ids"][idx]
        return image, caption

transform = transforms.Compose([
    transforms.Resize(img_shape[1:]),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageCaptionDataset(image_paths, tokenized_captions, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Vòng lặp đào tạo
for epoch in range(n_epochs):
    for i, (imgs, captions) in enumerate(dataloader):
        valid = torch.ones(imgs.size(0), 1).cuda()
        fake = torch.zeros(imgs.size(0), 1).cuda()

        real_imgs = imgs.cuda()
        
        # Đào tạo Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim).cuda()
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Đào tạo Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# Lưu mô hình
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
from safetensors.torch import save_file as save_safetensors
save_safetensors({"generator": generator.state_dict(), "discriminator": discriminator.state_dict()}, "gan_model.safetensors")

