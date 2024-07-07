import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from dataset import SegmentationDataset
from model import UNet
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, f1_score

class EarlyStopping:
    def __init__(self, patience=10, delta=0, save_path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_loss = float('inf')
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_val_loss = val_loss
        torch.save(model.state_dict(), self.save_path)

def train(model, device, train_loader, criterion, optimizer, epoch, epochs):
    model.train()
    running_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate batch accuracy
        preds = output > 0.5
        batch_correct = (preds == target).sum().item()
        batch_total = torch.numel(preds)
        batch_accuracy = 100 * batch_correct / batch_total

        # Accumulate epoch accuracy
        epoch_correct += batch_correct
        epoch_total += batch_total

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Batch Accuracy: {batch_accuracy:.2f}%')

    average_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * epoch_correct / epoch_total
    print(f"Train Epoch: {epoch} \tAverage Loss: {average_loss:.6f} \tEpoch Accuracy: {epoch_accuracy:.2f}%")

    # Log metrics to MLflow
    mlflow.log_metric('train_loss', average_loss, step=epoch)
    mlflow.log_metric('train_accuracy', epoch_accuracy, step=epoch)

def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

            # Calculate accuracy
            preds = output > 0.5
            correct += (preds == target).sum().item()
            total += torch.numel(preds)

            # Store all predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    average_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Average Loss: {average_loss:.6f} \tAccuracy: {accuracy:.2f}%')

    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()

    return average_loss, accuracy, all_targets, all_preds

def plot_confusion_matrix(all_targets, all_preds, epoch):
    cm = confusion_matrix(all_targets, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = cm_normalized * 100

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=['Primary', 'Background'])
    disp.plot(cmap=plt.cm.Blues, values_format='.2f')

    plt.title('Confusion Matrix (Percentages)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    cm_plot_path = f'confusion_matrix_epoch_{epoch}.png'
    plt.savefig(cm_plot_path)
    mlflow.log_artifact(cm_plot_path)
    plt.close()

def plot_f1_confidence_curve(all_targets, all_preds, epoch):
    precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_score = np.max(f1_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], 'b-', label='primary')
    plt.axvline(x=best_threshold, color='b', linestyle='-', label=f'all classes {best_f1_score:.2f} at {best_threshold:.3f}')
    plt.title('F1-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('F1')
    plt.ylim(0, 1)
    plt.legend()

    f1_plot_path = f'f1_confidence_curve_epoch_{epoch}.png'
    plt.savefig(f1_plot_path)
    mlflow.log_artifact(f1_plot_path)
    plt.close()

def main():
    data_path = '/Users/kieranmartin/Desktop/Spring 2024/Computer_Vision_Deep_Learning/Final Project/datasets/identdeforest'
    train_images_dir = os.path.join(data_path, 'train_seg_tagged/images')
    train_masks_dir = os.path.join(data_path, 'train_seg_tagged/masks')
    val_images_dir = os.path.join(data_path, 'val_seg_tagged/images')
    val_masks_dir = os.path.join(data_path, 'val_seg_tagged/masks')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # Initially added the augs below which made training more unstable and worse preformance
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(10)
    ])

    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform)
    val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # batch size > 4 results in worse preformance
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # Reduced batch size for small dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_classes=1).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced lr initially 0.001 then 0.005 lr
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)  # Cosine Annealing

    epochs = 200
    patience = 10

    experiment_name = 'UNet_Segmentation'
    mlflow.set_experiment(experiment_name)

    best_model_path = 'best_unet_modelv2.pth'
    early_stopping = EarlyStopping(patience=patience, save_path=best_model_path)

    with mlflow.start_run() as run:
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', 4) 
        mlflow.log_param('learning_rate', 0.0001) 
        mlflow.log_param('data_path', data_path)

        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, criterion, optimizer, epoch, epochs)
            val_loss, val_accuracy, _, _ = validate(model, device, val_loader, criterion)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_accuracy, step=epoch)

            early_stopping(val_loss, model)
            scheduler.step()

            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch}')
                break

        # Load the best model for evaluation
        model.load_state_dict(torch.load(best_model_path))
        _, _, all_targets, all_preds = validate(model, device, val_loader, criterion)

        # Plot confusion matrix and F1-confidence curve for the best model
        plot_confusion_matrix(all_targets, all_preds, 'best')
        plot_f1_confidence_curve(all_targets, all_preds, 'best')

        model_path = 'unet_model_finalv2.pth'
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()

