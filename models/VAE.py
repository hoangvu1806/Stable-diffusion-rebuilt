import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
import os, random, logging
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=256):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1), # for 3 layers
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1), # for 4 layers
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 16, kernel_size=4, stride=2, padding=1), # for 5 layers
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(hidden_dim * 4 * 64 * 64, latent_dim * 2)  # for 3 layers
            # nn.Linear(hidden_dim * 8 * 32 * 32, latent_dim * 2)  # for 4 layers
            nn.Linear(hidden_dim * 16 * 16 * 16, latent_dim * 2)  # mean and log-variance # for 5 layers
        )

        # Decoder
        self.decoder = nn.Sequential(
            # nn.Linear(latent_dim, hidden_dim * 4 * 64 * 64), # for 3 layers
            # nn.Linear(latent_dim, hidden_dim * 8 * 32 * 32), # for 4 layers
            nn.Linear(latent_dim, hidden_dim * 16 * 16 * 16), # for 5 layers
            nn.ReLU(),
            # nn.Unflatten(1, (hidden_dim * 4, 64, 64)), # for 3 layers
            # nn.Unflatten(1, (hidden_dim * 8, 32, 32)), # for 4 layers
            nn.Unflatten(1, (hidden_dim * 16, 16, 16)), # for 5 layers
            nn.ConvTranspose2d(hidden_dim * 16, hidden_dim * 8, kernel_size=4, stride=2, padding=1), # for 5 layers
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1), # for 4 layers
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1), # for 3 layers
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output is in range [-1, 1]
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mean, logvar = torch.chunk(encoded, 2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar

    def loss_function(self, recon_x, x, mean, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        # BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return (1.0*MSE)  + (1.0*KLD)
    
def train_vae(vae, data_dir, epochs=10, lr=0.003, batch_size=32, device=torch.device('cuda'), save_path='./saved_models'):
    log_path = os.path.join(save_path, "log.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    transform = transforms.Compose([
        # transforms.Resize((512, 512)),  # Thay đổi kích thước ảnh nếu cần
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load all images directly (ignoring subfolders)
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)

    # Convert to tensor and add batch dimension
    data = torch.stack(images)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=5e-3, step_size_up=2000, mode='triangular2')
    vae.to(device)
    losses = []  # List to store the loss values
    for epoch in tqdm(range(epochs),desc="training"):
        vae.train()
        train_loss = 0
        for batch_idx, data in tqdm(enumerate(dataloader)):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mean, logvar = vae(data)
            loss = vae.loss_function(recon_batch, data, mean, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()
        avg_loss = train_loss / len(dataloader.dataset)
        losses.append(avg_loss)  # Append the average loss for this epoch
        logging.info(f'Epoch {epoch+1}, Loss: {avg_loss}')
        # Lưu model sau mỗi 10 epoch
        if (epoch+1) % 10 == 0:
            torch.save(vae.state_dict(), f"{save_path}/VAE_3-64-64_face5LE0{epoch+1}.pth")
        print(f'====> Epoch {epoch+1} Average loss: {avg_loss}')
    # Plot the loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.show()

def load_vae(model_path, input_dim=3, hidden_dim=64, latent_dim=128, device=torch.device('cuda')):
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    # model.eval()
    return model

# Set the seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    SEED = 1806
    set_seed(SEED)  # Set the seed to any desired value
    input_dim = 3  # RGB image
    hidden_dim = 64
    latent_dim = 64
    device = torch.device('cuda')
    model_path = 'E:\Stable_Diffusion_rebuild\saved_models\VAE_3-64-64_face5LE010.pth'

    gen_mode = 1 # to swap train and test modes "gen_mode == False is training mode"
# =================================================================
    if gen_mode: # training model VAE
        # Load the model
        loaded_model = load_vae(model_path, input_dim, hidden_dim, latent_dim)
        loaded_model.eval()
        # Test the loaded model with an example image
        example_image_path = "E:\Stable_Diffusion_rebuild\datasets\images_resized\image_00223.jpg"
        example_image_path = r"E:\Stable_Diffusion_rebuild\datasets\faces\16000\image_00004.jpg"

        image = Image.open(example_image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        image = transform(image).unsqueeze(0).to(torch.device('cuda'))

        with torch.no_grad():
            reconstructed, mean, logvar = loaded_model(image)
            loss = loaded_model.loss_function(reconstructed, image, mean, logvar)
            print(f"Loss between original and reconstructed image: {loss.item()}")
        
        # Display the original and reconstructed image
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image.cpu().squeeze(0).permute(1, 2, 0).numpy())
        axs[0].set_title("Original Image")
        reconstructed_image = (reconstructed.cpu().squeeze(0).permute(1, 2, 0).numpy() + 1) / 2
        reconstructed_image = np.clip(reconstructed_image, 0, 1)
        print(reconstructed_image.shape)
        plt.figtext(0.5, 0.01, f'Loss: {loss.item()}', ha='center', fontsize=12)
        axs[1].imshow(reconstructed_image)
        axs[1].set_title("Reconstructed Image")
        plt.show()

# =================================================================
    else:
        data_dir = r"E:\Stable_Diffusion_rebuild\datasets\faces\16000"
        lr = 0.001
        epochs = 20
        batch_size = 20
        # model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        model = load_vae(model_path, input_dim, hidden_dim, latent_dim)
        train_vae(model, data_dir, epochs, lr, batch_size)
# =================================================================


