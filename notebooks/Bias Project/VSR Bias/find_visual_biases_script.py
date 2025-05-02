import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt


from datasets import load_from_disk
import os
from dotenv import load_dotenv
load_dotenv()

disk_loc = os.path.join(os.environ["DATA_PATH"], "vsr")


from datasets import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        
        mlp_dim = int(dim * mlp_ratio)

        self.mlp_layers = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        
        return self.mlp_layers(x)
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        '''
        Self-attention layer.
        
        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
        '''
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # TODO: Define here the linear layers producing K, Q, V from the input x

        # We map to full dim and we will reshape later to split the heads (but doing as so doesn't change the model)
        self.K = nn.Linear(dim, dim, bias=False)
        self.Q = nn.Linear(dim, dim, bias=False)

        # The paper uses d_k = d_v, so we will do the same
        self.V = nn.Linear(dim, dim, bias=False)
        
        # Projection
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        '''
        Performs a forward pass through the multi-headed self-attention layer.
        
        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.
            
        returns:
            Output of shape [B N C].
        '''
        B, N, C = x.shape
        
        # TODO: Compute the keys K, queries Q, and values V from x. Each should be of shape [B num_heads N head_dim].
        q = self.Q(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        k = self.K(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        v = self.V(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        # TODO: Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads N N].
        attn = q @ k.transpose(-2, -1) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n1 n2 -> b 1 n1 n2")
            # TODO: Apply the optional attention mask. Wherever the mask is True, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill(mask, float('-inf'))

        # TODO: Compute the softmax over the last dimension
        attn = F.softmax(attn, dim=-1)

        # TODO: Weight the values V by the attention matrix and concatenate the different attention heads
        x = attn @ v
        x = x.reshape(B, N, -1)
        
        # One final projection
        x = self.proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        '''
        Transformer encoder block.
        
        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
            :mlp_ratio: MLP hidden dimensionality multiplier
        '''
        super().__init__()
        
        self.layer_norm_x = nn.LayerNorm(dim)
        self.att = Attention(dim, num_heads)

        self.layer_norm_xa = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x, mask=None):
        '''
        Performs a forward pass through the multi-headed self-attention layer.
        
        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.
            
        returns:
            Output of shape [B N C].
        '''
        
        X_a = self.att(self.layer_norm_x(x), mask) + x
        X_b = self.mlp(self.layer_norm_xa(X_a)) + X_a

        return X_b
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=14, patch_size=2, in_channels=1, embed_dim=192):
        '''
        Image to Patch Embedding.
        
        params:
            :img_size: Image height and width in pixels
            :patch_size: Patch size height and width in pixels
            :in_channels: Number of input channels
            :embed_dim: Token dimension
        '''
        super().__init__()
        
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        '''
        Performs a forward pass through the patch embedding.
        
        params:
            :x: Input of shape [B C H W]. B = batch size, C = number of channels, H = image height, W = image width
            
        returns:
            Output of shape [B N C].
        '''
        y = self.projection(x)
        y = rearrange(y, "b c h w -> b (h w) c")

        return y

def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d
    Returns positional embedding of shape [1, N, D]
    """
    arange = torch.arange(max_len, dtype=torch.float32)
    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out = torch.einsum('m,d->md', [arange, omega])
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)[None, :, :]
    return pos_emb

def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3
    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    Returns positional embedding of shape [B, N, D]
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    return pos_emb


class ViT(nn.Module):
    
    def __init__(self, 
                 img_size=14, 
                 patch_size=2, 
                 in_channels=1, 
                 embed_dim=192, 
                 num_classes=10, 
                 depth=4, 
                 num_heads=4, 
                 mlp_ratio=4., 
                 **kwargs):
        '''
        A Vision Transformer for classification.
        
        params:
            :img_size: Image height and width in pixels
            :patch_size: Patch size height and width in pixels
            :in_channels: Number of input channels
            :embed_dim: Token dimension
            :num_classes: Number of classes
            :depth: Transformer depth
            :num_heads: Number of attention heads
            :mlp_ratio: MLP hidden dimensionality multiplier
        '''
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            build_2d_sincos_posemb(img_size//patch_size, img_size//patch_size, embed_dim=embed_dim), 
            requires_grad=False
        )
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes, bias=False)
            
    def forward(self, x):
        '''
        Forward pass through the ViT.
        
        params:
            :x: Input of shape [B C H W]. B = batch size, C = number of channels, H = image height, W = image width
            
        returns:
            Output of shape [B num_classes]
        '''        
        # TODO: Project images to patches
        x = self.patch_embed(x)
        
        # TODO: Add the positional embeddings to the tokens
        x += self.pos_embed
            
        # TODO: Forward pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # TODO: Perform average pooling (compute the mean over the sequences)
        x = x.mean(dim=1)
        
        # TODO: Compute the logits
        x = self.head(x)

        return x

def main():
    image_size = 48
    batch_size = 64
    num_workers = 8

    dataset = load_from_disk(disk_loc)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),          # → float tensor in [0,1]
    ])

    def add_pixel_values(example):
        img = Image.open(example["image_path"]).convert("RGB")
        example["pixel_values"] = transform(img)
        return example

    for split in dataset.keys():
        dataset[split] = dataset[split].map(
            add_pixel_values,
            remove_columns=["image_path"],  # drop once it’s inside pixel_values
            num_proc=num_workers,
        )

    # ------------------------------------------------------------------
    # 6. Tell datasets to yield PyTorch tensors
    # ------------------------------------------------------------------
    dataset.set_format(type="torch",
                    columns=["pixel_values", "label"], output_all_columns=False)  # include any columns you need


    loader_train = DataLoader(dataset['train'], batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    loader_val = DataLoader(dataset['validation'], batch_size=batch_size, num_workers=num_workers, drop_last=False)
    loader_test = DataLoader(dataset['test'], batch_size=batch_size, num_workers=num_workers, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    vit = ViT(
        img_size=image_size, patch_size=2, in_channels=3, 
        embed_dim=792, num_classes=2, depth=16, 
        num_heads=8, mlp_ratio=4., 
    ).to(device)
    optimizer = torch.optim.AdamW(vit.parameters())
    num_parameters = sum([p.numel() for p in vit.parameters()])
    print(f'Number of parameters: {num_parameters:,}')


    num_epochs = 20

    train_losses = []
    val_losses = []
    val_accuracies = []

    for _ in range(num_epochs):
            
        # Train loop
        vit.train()
        epoch_loss_train = 0
        pbar = tqdm(total=len(loader_train))
        for batch in loader_train:        
            
            inputs, targets = batch['pixel_values'].to(device), batch['label'].to(device)

            logits = vit(inputs)
            loss = F.cross_entropy(logits, targets)

            vit.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            epoch_loss_train += loss.item()
            
            pbar.update(1)
            pbar.set_description(f'Train loss: {loss.item():.3f}')
        pbar.close()
        
        epoch_loss_train /= len(loader_train)
        train_losses.append(epoch_loss_train)
        
        
        # Validation loop
        vit.eval()
        epoch_loss_val = 0
        correct = 0
        for batch in loader_val:
            inputs, targets = batch['pixel_values'].to(device), batch['label'].to(device)

            with torch.no_grad():
                logits = vit(inputs)
            loss = F.cross_entropy(logits, targets)
            
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            
            epoch_loss_val += loss.item()
            
        accuracy = correct / len(loader_val.dataset)
        
        epoch_loss_val /= len(loader_val)
        val_losses.append(epoch_loss_val)
        val_accuracies.append(accuracy)
        
        print(f'Train loss: {epoch_loss_train:.3f}, val loss: {epoch_loss_val:.3f}, val accuracy: {accuracy:.3f}')


    plt.figure()
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('losses.png')
    plt.show()

    test_loss = 0
    correct = 0

    vit.eval()
    for batch in loader_val:
        inputs, targets = batch['pixel_values'].to(device), batch['label'].to(device)

        with torch.no_grad():
            logits = vit(inputs)
        loss = F.cross_entropy(logits, targets)
        test_loss += loss.item()
        
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(loader_val)
    accuracy = correct / len(loader_val.dataset)

    print(f'Validation loss: {test_loss:.3f}')
    print(f'Validation top-1 accuracy: {accuracy*100}%')

    # Save on disk
    torch.save(vit.state_dict(), "vit.pth")


if __name__ == "__main__":
    main()