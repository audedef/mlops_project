# -*- coding: utf-8 -*-
"""
PyTorch Image Classification Training Script (Dandelion vs Grass)

This script trains a PyTorch image classification model using images stored in an S3 bucket (like MinIO) and labels from a CSV file.
It loads images directly from S3 during training using a custom Dataset and saves the final trained model state dictionary back to the S3 bucket.

Environment Variables :
For more security, create a .env file in the directory we run this from.
"""

from tqdm import tqdm  # For progress bars
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet34
from PIL import Image
from datetime import datetime
from pathlib import Path
import io
import os
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import time

# --- Configuration ---

# S3/MinIO Configuration
S3_ENDPOINT_URL = os.getenv('AWS_S3_ENDPOINT', 'http://minio:9000')
S3_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
S3_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
S3_BUCKET_NAME = 'images-bucket'  # bucket name configured in docker-compose/MinIO

# Data Configuration
LOCAL_CSV_PATH = Path('./data/dataset.csv')

# Model Training Configuration
MODEL_ARCH = resnet34  # good starting point
IMAGE_SIZE = 224  # Standard size for many ResNets
BATCH_SIZE = 16  # Increase if we have more GPU memory
# Number of epochs for fine-tuning (increase for potentially better accuracy)
EPOCHS = 5
LEARNING_RATE = 0.001
# Number of parallel workers for data loading (adjust based on CPU cores)
NUM_WORKERS = 2
NUM_CLASSES = 2  # Dandelion vs Grass

# Output Configuration
# Define a local temporary path for the model state_dict export
LOCAL_MODEL_FILENAME = 'pytorch_grass_dandelion_temp.pth'
# save in current directory temporarily
LOCAL_MODEL_PATH = Path(f'./{LOCAL_MODEL_FILENAME}')

# Define the desired S3 key (path within the bucket) for the model state_dict
# Using a timestamp for basic versioning
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
S3_MODEL_KEY = f'models/pytorch_grass_dandelion/{TIMESTAMP}/model_state_dict.pth'
print(
    f"Model state_dict will be saved to S3 key: s3://{S3_BUCKET_NAME}/{S3_MODEL_KEY}")

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- S3 Connection Setup ---


def get_s3_client():
    """Creates and returns a boto3 S3 client."""
    print(f"Attempting to connect to S3/MinIO endpoint: {S3_ENDPOINT_URL}")
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY
        )
        # Test connection by listing buckets (optional, requires ListBuckets permission)
        s3_client.list_buckets()  # This can fail if MINIO_BROWSER=off or permissions are wrong
        print("Successfully connected to S3/MinIO.")
        return s3_client
    except NoCredentialsError:
        print("S3 credentials not found. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set (e.g., in .env file).")
        return None
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if "InvalidAccessKeyId" in str(e) or "SignatureDoesNotMatch" in str(e) or error_code == 'InvalidAccessKeyId':
            print(
                f"S3 Authentication Error: {e}. Check access key and secret key.")
        elif "could not connect" in str(e).lower() or "endpoint url" in str(e).lower():
            print(
                f"S3 Connection Error: {e}. Check S3_ENDPOINT_URL ({S3_ENDPOINT_URL}). Is MinIO running and accessible?")
        elif error_code == 'AccessDenied':
            print(
                f"S3 Permissions Error: {e}. Check if the provided keys have permission (e.g., ListBuckets).")
        else:
            print(
                f"An S3 ClientError occurred: {e} (Error Code: {error_code})")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during S3 connection: {e}")
        return None


s3_client = get_s3_client()

if not s3_client:
    print("Exiting due to S3 connection failure.")
    exit(1)  # Use non-zero exit code for errors


# --- Data Loading (CSV) ---
df = None
# Option 1: Load CSV from Local Path
if LOCAL_CSV_PATH is not None:
    if LOCAL_CSV_PATH.exists():
        print(f"Loading dataset labels from local CSV: {LOCAL_CSV_PATH}")
        try:
            df = pd.read_csv(LOCAL_CSV_PATH)
        except Exception as e:
            print(f"Error reading local CSV file {LOCAL_CSV_PATH}: {e}")
            exit(1)
    else:
        print(f"Error: Local CSV file not found at {LOCAL_CSV_PATH}")
        exit(1)
# Option 2: Load CSV from S3
elif LOCAL_CSV_PATH is None and 'S3_CSV_KEY' in locals():
    # (Loading logic same as FastAI version)
    try:
        print(
            f"Loading dataset labels from S3: s3://{S3_BUCKET_NAME}/{S3_CSV_KEY}")
        csv_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_CSV_KEY)
        csv_content = csv_obj['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        print("Successfully loaded CSV from S3.")
    except ClientError as e:
        print(
            f"Error loading CSV from S3 (s3://{S3_BUCKET_NAME}/{S3_CSV_KEY}): {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading CSV from S3: {e}")
        exit(1)
else:
    print("Error: No valid CSV data source configured.")
    exit(1)

if df is None or df.empty:
    print("Error: DataFrame is empty or failed to load. Exiting.")
    exit(1)

required_cols = ['image_path', 'label']
if not all(col in df.columns for col in required_cols):
    print(
        f"Error: CSV missing required columns. Found: {df.columns}. Required: {required_cols}")
    exit(1)

print(f"\nDataset loaded. Shape: {df.shape}")
print("Sample rows:")
print(df.head())
print("\nValue counts for 'label':")
print(df['label'].value_counts())


# --- Map Labels to Integers ---
# Create a mapping from label names to integers
unique_labels = sorted(df['label'].unique())
label_map = {label: i for i, label in enumerate(unique_labels)}
print(f"Label mapping: {label_map}")
if len(label_map) != NUM_CLASSES:
    print(
        f"Warning: Number of unique labels ({len(label_map)}) does not match NUM_CLASSES ({NUM_CLASSES}). Check data or config.")

df['label_int'] = df['label'].map(label_map)


# --- Custom Function to Load Images from S3 ---

def s3_parse_path(s3_path):
    """Parses an s3://bucket/key path."""
    if not isinstance(s3_path, str) or not s3_path.startswith("s3://"):
        raise ValueError(
            f"Invalid S3 path format: {s3_path} (Type: {type(s3_path)})")
    parts = s3_path[5:].split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    if not bucket or not key:
        raise ValueError(f"Could not parse bucket/key from S3 path: {s3_path}")
    return bucket, key


class S3ImageDataset(Dataset):
    """Custom PyTorch Dataset to load images directly from S3."""

    def __init__(self, dataframe, s3_client, transform=None):
        self.dataframe = dataframe
        self.s3_client = s3_client
        self.transform = transform
        self.bucket_cache = {}  # Basic cache for bucket names if needed

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        s3_path = row['image_path']
        label = row['label_int']  # Use the integer label
        image = None  # Initialize image to None

        try:
            bucket, key = s3_parse_path(s3_path)
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            img_bytes = obj['Body'].read()
            bytes_read = len(img_bytes)
            if not img_bytes:
                print(
                    f"WARNING: Worker {os.getpid()} skipping zero-byte file: {s3_path}")
                return None

            # 1. Open the image from bytes
            image_pil = Image.open(io.BytesIO(img_bytes))

            # 3. Convert to RGB (standardizes channels)
            image_pil = image_pil.convert('RGB')
            print(
                f"DEBUG: Worker {os.getpid()} successfully opened and converted {s3_path}")

            # 4. Apply transformations ONLY if image was loaded and converted successfully
            if self.transform:
                try:
                    # Apply transforms to the PIL image
                    image = self.transform(image_pil)
                except Exception as e:
                    print(
                        f"ERROR: Worker {os.getpid()} failed applying transform to image {s3_path} (index {idx}): {e}")
                    return None  # Return None if transform fails

            # If we got here, image should be a transformed tensor
            if image is None:  # Should not happen if transform didn't fail, but safety check
                print(
                    f"WARNING: Worker {os.getpid()} resulted in None image after transform for {s3_path}")
                return None
            else:
                return image, label

        # Catch specific exceptions first if possible
        except Image.UnidentifiedImageError:
            print(
                f"WARNING: Worker {os.getpid()} Pillow could not identify image file format for {s3_path}. Skipping.")
            return None
        except OSError as e:
            # Catch the "broken data stream" or other OS/IO errors during open/convert
            print(
                f"WARNING: Worker {os.getpid()} Pillow/IO OSError for {s3_path}. Skipping. Error: {e}")
            return None
        # Catch S3 or other general errors
        except (ClientError, ValueError, Exception) as e:
            print(
                f"WARNING: Worker {os.getpid()} failed loading/processing image {s3_path} for index {idx}: {e}")
            return None


def collate_fn_skip_corrupted(batch):
    """
    Custom collate_fn that filters out None values from a batch.
    These None values are expected if __getitem__ failed to load an image.
    """
    # Filter out None entries first
    original_size = len(batch)
    batch = [item for item in batch if item is not None]
    filtered_size = len(batch)

    if original_size > filtered_size:
        print(
            f"DEBUG: Collate filtered out {original_size - filtered_size} corrupted item(s).")

    # If the entire batch was corrupted...
    if not batch:
        # Return dummy tensors or signal somehow. Returning empty tensors is one way.
        # The shape should match what the rest of your loop expects.
        # Example: return torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE), torch.empty(0, dtype=torch.long)
        # Simpler: return None and handle in the loop
        return None

    # If batch is valid, use the default collate function
    return torch.utils.data.dataloader.default_collate(batch)


# --- Image Transformations ---
# Standard ImageNet normalization values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Define transforms for training (with augmentation) and validation (no augmentation)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        # Resize slightly larger than crop size
        transforms.Resize(IMAGE_SIZE + 16),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize
    ]),
}


# --- Split Data and Create DataLoaders ---
print("\nSplitting data into training and validation sets...")
try:
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,       # 20% for validation
        random_state=42,     # For reproducibility
        # Ensure class distribution is similar in both sets
        stratify=df['label_int']
    )
except ValueError as e:
    print(
        f"Error during train/val split (likely too few samples per class for stratification): {e}")
    print("Trying split without stratification...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Create Dataset instances
print("Creating PyTorch Datasets...")
try:
    train_dataset = S3ImageDataset(
        train_df, s3_client, transform=data_transforms['train'])
    val_dataset = S3ImageDataset(
        val_df, s3_client, transform=data_transforms['val'])
except Exception as e:
    print(f"Error creating Dataset instances: {e}")
    exit(1)

# Create DataLoader instances
print("Creating PyTorch DataLoaders...")
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_corrupted),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_corrupted)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}


# --- Model Definition (Transfer Learning) ---

print(f"\nLoading pre-trained model: {MODEL_ARCH}")
# Load the specified pre-trained model
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

# Modify the final classification layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
print(
    f"\nReplaced ResNet final layer (fc) with {NUM_CLASSES} output features.")

# Move model to the appropriate device (GPU or CPU)
model = model.to(device)

# --- Loss Function and Optimizer ---

criterion = nn.CrossEntropyLoss()  # Standard loss for multi-class classification

# Optimizer: For simplicity here, we optimize all parameters with the same LR.
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- Training Loop ---
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=EPOCHS):
    """Handles the training and validation loop."""
    since = time.time()
    best_model_wts = model.state_dict()  # Keep track of best weights
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Use tqdm for progress bar
            data_iterator = tqdm(
                dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")

            # Iterate over data FROM the DataLoader.
            # Assign the output of the iterator to 'batch_data' FIRST
            for batch_data in data_iterator:
                # Check if the collate_fn returned None (meaning the whole batch failed)
                if batch_data is None:
                    print("WARNING: Skipping empty/corrupted batch.")
                    continue  # Skip to the next iteration
                # If the batch is valid, unpack it
                inputs, labels = batch_data

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Get the predicted class index
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                running_loss += batch_loss
                running_corrects += batch_corrects

                # Update tqdm description
                data_iterator.set_postfix(
                    loss=batch_loss/inputs.size(0), acc=batch_corrects.double().item()/inputs.size(0))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(
                epoch_acc.item())  # Store accuracy as float

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print(
                    f'*** Best validation accuracy updated: {best_acc:.4f} ***')

    time_elapsed = time.time() - since
    print(
        f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


print("\nStarting training...")
# Train the model
model, history = train_model(
    model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=EPOCHS)


# --- Evaluation ---

print("\nFinal Evaluation on Validation Set...")
model.eval()  # Set model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(dataloaders['val'], desc="Final Validation"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
final_accuracy = accuracy_score(all_labels, all_preds)
print(f"\nFinal Validation Accuracy: {final_accuracy:.4f}")

print("\nClassification Report:")
# Map integer labels back to names for report
target_names = [name for name, idx in sorted(
    label_map.items(), key=lambda item: item[1])]
print(classification_report(all_labels, all_preds, target_names=target_names))


# --- Model Saving ---

print(f"\nSaving model state dictionary...")
print(f"Local temporary path: {LOCAL_MODEL_PATH}")
print(f"Target S3 path: s3://{S3_BUCKET_NAME}/{S3_MODEL_KEY}")

try:
    # 1. Save the model state dictionary locally
    torch.save(model.state_dict(), LOCAL_MODEL_PATH)
    print(f"Model state_dict saved locally successfully.")

    # 2. Upload the saved file to S3
    print(f"Uploading model state_dict to S3...")
    s3_client.upload_file(
        Filename=str(LOCAL_MODEL_PATH),  # boto3 needs string path
        Bucket=S3_BUCKET_NAME,
        Key=S3_MODEL_KEY
    )
    print(
        f"Model state_dict uploaded to S3 successfully: s3://{S3_BUCKET_NAME}/{S3_MODEL_KEY}")

    # 3. Clean up the local temporary file
    try:
        os.remove(LOCAL_MODEL_PATH)
        print(f"Removed local temporary model file: {LOCAL_MODEL_PATH}")
    except OSError as e:
        print(
            f"Warning: Could not remove local temporary file {LOCAL_MODEL_PATH}: {e}")

except FileNotFoundError:
    print(
        f"Error during save: Local path {LOCAL_MODEL_PATH.parent} not found or not writable.")
    exit(1)
except ClientError as e:
    print(f"Error uploading model state_dict to S3: {e}")
except Exception as e:
    print(f"An unexpected error occurred during model saving/upload: {e}")


print("\n--- Script Finished ---")
