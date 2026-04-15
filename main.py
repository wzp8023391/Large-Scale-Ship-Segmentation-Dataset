import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from skimage import io
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# from nnUnet import nnUNet
from module.ShipSegNet import ShipSegNet
# from Unet import SimpleUNet

class Config2:
    
    train_img_dir = r""  
    train_mask_dir = r""   
    val_img_dir = r""
    val_mask_dir = r""

    

    img_size = 512
    batch_size = 2
    epochs = 50
    lr = 1e-4
    weight_decay = 1e-5
    save_dir = 'Unet_pth'
    log_interval = 20          
    val_interval = 1          
    
    
    bce_weight = 0.3
    dice_weight = 0.7
    smooth = 1e-5
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    num_workers = 4
    seed = 42

cfg = Config2()


class ShipDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        

        self.img_names = sorted([os.path.basename(f) for f in glob(os.path.join(img_dir, '*.png'))])
        self.mask_names = sorted([os.path.basename(f) for f in glob(os.path.join(mask_dir, '*.png'))])
        
        assert len(self.img_names) == len(self.mask_names), "data error"
        
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        

        image = io.imread(img_path)
        if len(image.shape) == 2:  
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:  
            image = image[:, :, :3]
        

        mask = io.imread(mask_path, as_gray=True)

        mask = (mask > 0).astype(np.float32)
        

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # shape: (1, H, W)
        

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask


class BCEDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.3, dice_weight=0.7, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):

        

        pred_fore = pred[:, 1:2, :, :]   # shape: (B, 1, H, W)
        

        bce_loss = self.bce(pred_fore, target)
        

        pred_prob = torch.sigmoid(pred_fore)  
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        

        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_score(pred, target, smooth=1e-5):

    pred = (pred > 0.5).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

def save_checkpoint(model, optimizer, epoch, loss, best_dice, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_dice': best_dice,
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_dice = checkpoint['best_dice']
    return epoch, loss, best_dice


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = len(val_loader)
    
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)          # (B, 2, H, W)
        loss = criterion(outputs, masks) 
        
        # 计算Dice
        pred = torch.argmax(outputs, dim=1, keepdim=True)  # (B, 1, H, W)
        dice = dice_score(pred, masks)
        
        total_loss += loss.item()
        total_dice += dice
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    return avg_loss, avg_dice


def train():
    set_seed(cfg.seed)
    

    os.makedirs(cfg.save_dir, exist_ok=True)
    
    
    train_dataset = ShipDataset(cfg.train_img_dir, cfg.train_mask_dir)
    val_dataset = ShipDataset(cfg.val_img_dir, cfg.val_mask_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    

    model = ShipSegNet().to(cfg.device)
    

    criterion = BCEDiceLoss(bce_weight=cfg.bce_weight, dice_weight=cfg.dice_weight, smooth=cfg.smooth)
    

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    


    

    best_dice = 0.0
    start_epoch = 0

    # resume_path = os.path.join(cfg.save_dir, 'latest.pth')
    # if os.path.exists(resume_path):
    #     start_epoch, _, best_dice = load_checkpoint(model, optimizer, resume_path)
    #     print(f"Resumed from epoch {start_epoch}, best dice {best_dice:.4f}")
    
    
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(cfg.device)
            masks = masks.to(cfg.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            

            pred = torch.argmax(outputs, dim=1, keepdim=True)
            dice = dice_score(pred, masks)
            
            epoch_loss += loss.item()
            epoch_dice += dice
            

            pbar.set_postfix({'loss': loss.item(), 'dice': dice})
            

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = epoch_dice / len(train_loader)
        

        

        if (epoch + 1) % cfg.val_interval == 0:
            val_loss, val_dice = validate(model, val_loader, criterion, cfg.device)
            print(f" - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            
            

            if val_dice > best_dice:
                best_dice = val_dice
                best_path = os.path.join(cfg.save_dir, f'best_model.pth')
                save_checkpoint(model, optimizer, epoch+1, val_loss, best_dice, best_path)
                print(f"best_Dice: {val_dice:.4f} -> {best_path}")
            

            latest_path = os.path.join(cfg.save_dir, 'latest.pth')
            save_checkpoint(model, optimizer, epoch+1, val_loss, best_dice, latest_path)
    
    print(f"Dice: {best_dice:.4f}")


def inference_example(model_path, image_path):
    """加载模型并对单张图像进行预测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShipSegNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    

    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1,3,H,W)
    
    with torch.no_grad():
        output = model(image_tensor)  # (1,2,H,W)
        pred = torch.argmax(output, dim=1, keepdim=True)  # (1,1,H,W)
        pred = pred.squeeze().cpu().numpy()  # (H,W)
    
    # 保存预测结果
    io.imsave('prediction.png', (pred * 255).astype(np.uint8))
    print("save prediction.png")

def inference_folder(model_path, image_folder, out_folder):

    os.makedirs(out_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShipSegNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder,image_name)

        image = io.imread(image_path)
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)  # (1,3,H,W)
        
        with torch.no_grad():
            output = model(image_tensor)  # (1,2,H,W)
            pred = torch.argmax(output, dim=1, keepdim=True)  # (1,1,H,W)
            pred = pred.squeeze().cpu().numpy()  # (H,W)
        


        out_path = os.path.join(out_folder,image_name)
        io.imsave(out_path, (pred * 255).astype(np.uint8))

    print("Over")

if __name__ == '__main__':
    train()
    # out_path = 'ShipSegNet_result'
    # pth_path = r"ShipSegNet_pth\best_model.pth"
    # inference_folder(pth_path, r"", out_path)