import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
import glob

class FlatImageDataset(Dataset):
    """Custom dataset for loading images from a flat directory and assigning them a class label"""
    def __init__(self, root_dir, transform=None, class_label=0, class_name="Not_Found"):
        self.root_dir = root_dir
        self.transform = transform
        self.class_label = class_label
        self.class_name = class_name
        # Get all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        self.image_paths = sorted(self.image_paths)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.class_label
        except Exception as e:
            # If image fails to load, return a black image
            image = Image.new('RGB', (128, 128), color='black')
            if self.transform:
                image = self.transform(image)
            return image, self.class_label

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10  # Increased epochs for better training
    image_size = 128

    # Dataset paths
    unsplash_dir = "./Final_Spinach_NPK_Dataset_Clean/unsplash-images-collection 2"
    original_train_dir = "./Final_Spinach_NPK_Dataset_Clean/train"
    val_dir = "./Final_Spinach_NPK_Dataset_Clean/val"
    
    # Check if directories exist
    if not os.path.exists(unsplash_dir):
        print(f"Error: Unsplash directory not found at {unsplash_dir}")
        return
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return

    # Data Transforms with Augmentation for training
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load validation dataset first to get existing classes
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transforms)
    
    if len(val_dataset.classes) == 0 or len(val_dataset) == 0:
        print(f"Error: No classes or images found in {val_dir}")
        print(f"Found {len(val_dataset.classes)} classes and {len(val_dataset)} images")
        return
    
    # Get existing classes from validation set
    existing_classes = list(val_dataset.classes)
    print(f"\nExisting classes from validation set: {existing_classes}")
    
    # Load unsplash images as "Not_Found" class
    print(f"\nLoading unsplash images from: {unsplash_dir}")
    unsplash_dataset = FlatImageDataset(
        root_dir=unsplash_dir, 
        transform=train_transforms, 
        class_label=len(existing_classes),  # Assign new class index
        class_name="Not_Found"
    )
    
    if len(unsplash_dataset) == 0:
        print(f"Error: No images found in {unsplash_dir}")
        return
    
    print(f"Loaded {len(unsplash_dataset)} images as 'Not_Found' class")
    
    # Try to load original training data if it exists
    train_datasets = [unsplash_dataset]  # Start with unsplash images
    
    if os.path.exists(original_train_dir):
        try:
            original_train = torchvision.datasets.ImageFolder(root=original_train_dir, transform=train_transforms)
            if len(original_train.classes) > 0 and len(original_train) > 0:
                print(f"Found original training data: {len(original_train)} images with classes {original_train.classes}")
                train_datasets.append(original_train)
        except Exception as e:
            print(f"Could not load original training data: {e}")
    
    # Combine datasets
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        # Classes: existing + Not_Found
        classes = existing_classes + ["Not_Found"]
        print(f"\nCombined training datasets:")
        print(f"  - Original training: {len(train_datasets[1])} images")
        print(f"  - Unsplash (Not_Found): {len(unsplash_dataset)} images")
    else:
        # Only unsplash images - this won't work well for multi-class classification
        print(f"\n⚠️  WARNING: Only using unsplash images as training data!")
        print(f"⚠️  This will create a single-class model which won't work for classification.")
        print(f"⚠️  Consider adding the original training data for proper multi-class training.")
        train_dataset = unsplash_dataset
        classes = ["Not_Found"]
    
    print(f"\nTotal training images: {len(train_dataset)}")
    print(f"Classes: {classes} ({len(classes)} classes)")

    # Use num_workers=0 on macOS to avoid multiprocessing issues, or 2 on Linux/Windows
    import platform
    num_workers = 0 if platform.system() == 'Darwin' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Classes are already set above
    print(f"\n{'='*50}")
    print(f"Dataset Information:")
    print(f"{'='*50}")
    print(f"Classes: {classes}")
    print(f"Number of classes: {len(classes)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Note: Validation set has {len(val_dataset.classes)} classes, model will predict {len(classes)} classes")
    print(f"{'='*50}\n")
    
    # Save classes to JSON
    with open("classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    print(f"Classes saved to classes.json\n")

    # Simple CNN Model
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * (image_size // 8) * (image_size // 8), 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleCNN(num_classes=len(classes)).to(device)
    print(model)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Lists to keep track of progress
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    # Training Loop
    print("Starting training...")
    print(f"{'='*50}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        # Per-class accuracy tracking (only for classes in validation set)
        val_class_names = val_dataset.classes
        class_correct = {cls: 0 for cls in val_class_names}
        class_total = {cls: 0 for cls in val_class_names}
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # For validation, we only care about the original classes
                # Map validation labels to model output (they should match for first 4 classes)
                val_loss = criterion(outputs[:, :len(val_class_names)], labels)  # Only compute loss on existing classes
                val_running_loss += val_loss.item()
                
                # Get predictions (only consider original classes for validation accuracy)
                _, predicted = torch.max(outputs[:, :len(val_class_names)].data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Per-class accuracy (only for validation classes)
                for j in range(labels.size(0)):
                    label = labels[j].item()
                    if label < len(val_class_names):
                        class_name = val_class_names[label]
                        class_total[class_name] += 1
                        if predicted[j] == labels[j]:
                            class_correct[class_name] += 1
        
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        print(f"  Per-class Val Accuracy:")
        for cls in val_class_names:
            if class_total[cls] > 0:
                cls_acc = 100 * class_correct[cls] / class_total[cls]
                print(f"    {cls}: {cls_acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")
        if "Not_Found" in classes:
            print(f"    Not_Found: N/A (not in validation set)")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "model.pth")
            print(f"  ✓ Best model saved! (Val Acc: {best_val_acc:.2f}%)")
        
        print(f"{'='*50}\n")

    # Final model save (best model already saved during training)
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final model saved to model.pth")
    print(f"{'='*50}\n")

    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(epochs, val_losses, label='Val Loss', color='red', alpha=0.7)
    plt.plot(epochs, [acc/100 for acc in val_accuracies], label='Val Acc (scaled)', color='green', alpha=0.7)
    plt.title('Training Progress Overview')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
    print("Training metrics saved to training_metrics.png")

if __name__ == "__main__":
    main()
