import os
import time
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
from torchvision import transforms
from .metrics import classification_metrics, l2_normalize, verification_metrics_report, calculate_cosine_similarity, calculate_euclid_distance
from .net import IResNetEncoder, MyModel
from .loss_fn import BatchHardTripletLoss

def load_model(type="iresnet", model_size=18, embedding_dim=512, dropout_rate=0.4, sd_path=None):
    if type is None or type not in ["iresnet", "base"]:
        raise ValueError("Unknow your type")
    if type == "iresnet":
        if model_size is None or model_size not in [18, 34, 50, 100, 200]:
            raise ValueError("Your type is iresnet, so model size must be in [18, 34, 50, 100, 200]")
        else:
            model = IResNetEncoder(model_size, embedding_dim, dropout_rate)
    else:
        model = MyModel(embedding_dim, dropout_rate)
    if sd_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(sd_path, map_location=device)
        model.load_state_dict(sd["model"])
    return model

def create_data_splits(dataset, val_factor):
    length = len(dataset)
    val_size = int(length * val_factor)
    train_size = length - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def crop_face(img_path, threshold=0.8):
    # Dùng YOLO đã train để crop face
    from ultralytics import YOLO

    model = YOLO(r"D:\private\face_recognition\face_detection\runs\detect\yolov10n_640\weights\best.pt")
    print("Load model YOLO thành công")
    results = model.predict(img_path)
    bboxes = []
    for res in results:
        conf = max(res.boxes.conf)
        if conf >= threshold:
            xyxy = res.boxes.xyxy
            bboxes.append(xyxy[0].tolist())
    img = Image.open(img_path)
    img = img.crop(bboxes[0])
    img.show()
    return img


def define_transform():
    # train_transform = transforms.Compose([transforms.Resize((112, 112)),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15,
        hue=0.03,
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.1),

    transforms.RandomGrayscale(p=0.05),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
    ),
])
    val_transform = transforms.Compose([transforms.Resize((112, 112)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return train_transform, val_transform


def train(model, train_loader, val_loader, epochs, optimizer, loss_fn, save_path, device, scheduler, num_thresholds: int=400, target_at_far=0.01, epsilon=1e-6):
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, "checkpoints")
    report_path = os.path.join(save_path, "reports")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(report_path, exist_ok=True)

    best_save_path = os.path.join(checkpoint_path, "best.pth")
    last_save_path = os.path.join(checkpoint_path, "last.pth")
    his_save_path = os.path.join(report_path, "history.json")

    history = {"train_loss": [], "val_loss": [], "train_precision": [], "val_precision": [],
               "train_recall": [], "val_recall": [], "train_f1": [], "val_f1": [], "train_acc": [], "val_acc": [],
               "eer": [], "eer_threshold": [], "tar_at_far": [], "far_at_target": [], "threshold_at_target_far": [], "auc": [], "learning_rate": []}

    is_triplet_loss = isinstance(loss_fn, BatchHardTripletLoss)
    loss_fn = loss_fn.to(device)
    model = model.to(device)
    best_score = float("inf")
    total_time = 0.0
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        loss_fn.train()
        train_running_loss = 0
        if not is_triplet_loss:
            train_preds, train_labels = [], []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", leave=False)
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embedding = model(images)
            loss = loss_fn(embedding, labels)
            loss.backward()
            optimizer.step()
            if not is_triplet_loss:
                with torch.no_grad():
                    w = loss_fn.W.detach().cpu().numpy()
                    embedding_np = embedding.detach().cpu().numpy()
                    w_norm = l2_normalize(w, epsilon)
                    emb_norm = l2_normalize(embedding_np, epsilon)
                    preds = np.argmax(np.dot(emb_norm, w_norm.T), axis=1)
                    train_preds.extend(preds.tolist())
                    train_labels.extend(labels.detach().cpu().tolist())
            train_running_loss += loss.item()

        train_epoch_loss = train_running_loss / len(train_loader)
        if not is_triplet_loss:
            train_metrics = classification_metrics(train_labels, train_preds)
            train_acc = train_metrics["accuracy"]
            train_precision = train_metrics["precision"]
            train_recall = train_metrics["recall"]
            train_f1 = train_metrics["f1"]

        model.eval()
        loss_fn.eval()
        val_running_loss = 0
        if not is_triplet_loss:
            val_preds = []
        val_labels = []
        val_embeddings = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validating]", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                embedding = model(images)
                loss = loss_fn(embedding, labels)
                val_running_loss += loss.item()

                if not is_triplet_loss:
                    w = loss_fn.W.detach().cpu().numpy()
                    embedding_np = embedding.detach().cpu().numpy()
                    w_norm = l2_normalize(w, epsilon)
                    emb_norm = l2_normalize(embedding_np, epsilon)
                    preds = np.argmax(np.dot(emb_norm, w_norm.T), axis=1)
                    val_preds.extend(preds.tolist())
                val_labels.extend(labels.detach().cpu().tolist())
                val_embeddings.append(embedding.detach().cpu())

        val_epoch_loss = val_running_loss / len(val_loader)
        if not is_triplet_loss:
            val_metrics = classification_metrics(val_labels, val_preds)
            val_acc = val_metrics["accuracy"]
            val_precision = val_metrics["precision"]
            val_recall = val_metrics["recall"]
            val_f1 = val_metrics["f1"]
        val_embeddings = torch.cat(val_embeddings, dim=0)
        results = verification_metrics_report(val_embeddings, val_labels, num_thresholds=num_thresholds, target_at_far=target_at_far)

        end = time.perf_counter()
        epoch_time = (end - start) / 60
        total_time += epoch_time

        tar_at_far = results["tar_at_far"]
        eer_results = results["metrics_at_eer"]
        auc = results["auc"]
        current_eer = eer_results["eer"]
        print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.4f}m: TrLoss={train_epoch_loss:.4f} | ValLoss={val_epoch_loss:.4f}")
        if not is_triplet_loss:
            print(f"    - Classification(TrAcc={train_acc:.4f} TrP={train_precision:.4f} TrR={train_recall:.4f} TrF1={train_f1:.4f} | ValAcc={val_acc:.4f} ValP={val_precision:.4f} ValR={val_recall:.4f} ValF1={val_f1:.4f})")
        print(f"    - Verification(TAR@FAR{target_at_far}={tar_at_far['tar@target_far']:.4f} FAR={tar_at_far['far@target_far']:.4f} Threshold={tar_at_far['threshold@target_far']:.4f} | EER={current_eer:.4f} EERThreshold={eer_results['threshold@eer']:.4f} AUC={auc:.4f})")

        checkpoints = {"model": model.state_dict(),
                       "loss_fn": loss_fn.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "epoch": epoch}
        if current_eer < best_score:
            best_score = current_eer
            torch.save(checkpoints, best_save_path)
        torch.save(checkpoints, last_save_path)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(best_score)
            else:
                scheduler.step()

        # Save history
        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        if not is_triplet_loss:
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_precision"].append(train_precision)
            history["val_precision"].append(val_precision)
            history["train_recall"].append(train_recall)
            history["val_recall"].append(val_recall)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)
        history["eer"].append(current_eer)
        history["eer_threshold"].append(eer_results["threshold@eer"])
        history["tar_at_far"].append(tar_at_far["tar@target_far"])
        history["far_at_target"].append(tar_at_far["far@target_far"])
        history["threshold_at_target_far"].append(tar_at_far["threshold@target_far"])
        history["auc"].append(auc)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

    history["total_time"] = total_time
    with open(his_save_path, "w") as f:
        json.dump(history, f)
    print(f"History is saved")
    print(f"Training completely with {total_time:.2f} minutes!")
    return history


def plot_history(history, sp=None):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    p_train = history["train_precision"]
    p_val = history["val_precision"]
    r_train = history["train_recall"]
    r_val = history["val_recall"]
    f1_train = history["train_f1"]
    f1_val = history["val_f1"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    eer = history["eer"]
    far = history["far_at_target"]
    tar = history["tar_at_far"]
    auc = history["auc"]
    figs = []
    epochs = [i + 1 for i in range(len(train_loss))]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Loss
    idx = np.argmin(val_loss)
    min_epoch = epochs[idx]
    min_val = val_loss[idx]
    ax[0].plot(epochs, train_loss, label="Train Loss")
    ax[0].plot(epochs, val_loss, label="Val Loss")
    ax[0].annotate(text=f"Min Val Loss at\n(Epoch: {min_epoch}, Loss: {min_val:.4f})",
                   xy=(min_epoch, min_val), textcoords="offset points",
                   xytext=(20, 20), arrowprops=dict(arrowstyle="->", color="red"),
                   fontsize=10, color="red")
    ax[0].set_title("Training Loss & Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Accuracy
    idx = np.argmax(val_acc)
    best_epoch = epochs[idx]
    best_val = val_acc[idx]
    ax[1].plot(epochs, train_acc, label="Train Acc")
    ax[1].plot(epochs, val_acc, label="Val Acc")
    ax[1].scatter(best_epoch, best_val, s=50)
    ax[1].annotate(
        f"Best Val Acc\n({best_epoch}, {best_val:.4f})",
        xy=(best_epoch, best_val),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red")
    ax[1].set_title("Training Accurcay & Validation Accurcay")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Precision
    idx = np.argmax(p_val)
    best_epoch = epochs[idx]
    best_val = p_val[idx]
    ax[0].plot(epochs, p_train, label="Train Precision")
    ax[0].plot(epochs, p_val, label="Val Precision")
    ax[0].scatter(best_epoch, best_val, s=50)
    ax[0].annotate(
        f"Best Val Precision\n({best_epoch}, {best_val:.4f})",
        xy=(best_epoch, best_val),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red")
    ax[0].set_title("Training Precision & Validation Precision")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Precision")
    ax[0].legend()

    # Recall
    idx = np.argmax(r_val)
    best_epoch = epochs[idx]
    best_val = r_val[idx]
    ax[1].plot(epochs, r_train, label="Train Recall")
    ax[1].plot(epochs, r_val, label="Val Recall")
    ax[1].scatter(best_epoch, best_val, s=50)
    ax[1].annotate(
        f"Best Val Recall\n({best_epoch}, {best_val:.4f})",
        xy=(best_epoch, best_val),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red")
    ax[1].set_title("Training Recall & Validation Recall")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Recall")
    ax[1].legend()

    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # F1
    idx = np.argmax(f1_val)
    best_epoch = epochs[idx]
    best_val = f1_val[idx]
    ax.plot(epochs, f1_train, label="Train F1")
    ax.plot(epochs, f1_val, label="Val F1")
    ax.scatter(best_epoch, best_val, s=50)
    ax.annotate(
        f"Best Val F1\n({best_epoch}, {best_val:.4f})",
        xy=(best_epoch, best_val),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red")
    ax.set_title("Training F1 & Validation F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.legend()

    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # EER
    idx = np.argmin(eer)
    best_epoch = epochs[idx]
    best_eer = eer[idx]
    ax[0].plot(epochs, eer, label="EER")
    ax[0].annotate(
        f"Best EER\n({best_epoch}, {best_eer:.4f})",
        xy=(best_epoch, best_eer),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red")
    ax[0].set_title("EER during training")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("EER values")
    ax[0].legend()

    # FAR
    idx = np.argmin(far)
    best_epoch = epochs[idx]
    best_far = far[idx]
    ax[1].plot(epochs, far, label="FAR")
    ax[1].annotate(
        f"Best FAR\n({best_epoch}, {best_far:.4f})",
        xy=(best_epoch, best_far),
        xytext=(21, 21),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red")
    ax[1].set_title("FAR during training")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("FAR values")
    ax[1].legend()

    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # TAR
    idx = np.argmax(tar)
    best_epoch = epochs[idx]
    best_tar = tar[idx]
    ax[0].plot(epochs, tar, label="TAR")
    ax[0].annotate(
        f"Best TAR\n({best_epoch}, {best_tar:.4f})",
        xy=(best_epoch, best_tar),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red")
    ax[0].set_title("TAR during training")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("TAR values")
    ax[0].legend()

    # AUC
    idx = np.argmax(auc)
    best_epoch = epochs[idx]
    best_auc = auc[idx]
    ax[1].plot(epochs, auc, label="AUC")
    ax[1].annotate(
        f"Best AUC\n({best_epoch}, {best_auc:.4f})",
        xy=(best_epoch, best_auc),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red")
    ax[1].set_title("AUC during training")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("AUC values")
    ax[1].legend()

    fig.tight_layout()
    figs.append(fig)
    if sp is not None:
        sp = os.path.join(sp, "reports")
        os.makedirs(sp, exist_ok=True)
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(sp, f"{i}.png"))
    plt.show()


def face_verification(img_path1, img_path2, model, mode="cosine"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([transforms.Resize((112, 112)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img1_cropped = crop_face(img_path=img_path1).convert("RGB")
    img2_cropped = crop_face(img_path=img_path2).convert("RGB")
    img1 = trans(img1_cropped)
    img2 = trans(img2_cropped)
    img1 = img1.to(device)
    img2 = img2.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        img1_emb = model(img1.unsqueeze(0)).cpu()
        img2_emb = model(img2.unsqueeze(0)).cpu()
        if mode == "cosine":
            result = calculate_cosine_similarity(img1_emb, img2_emb)
        elif mode == "euclid":
            result = calculate_euclid_distance(img1_emb, img2_emb)
    return result, img1_emb, img2_emb


if __name__ == "__main__":
    history_test = {
    "train_loss": [
        32.12, 29.43, 26.85, 23.72, 20.54,
        17.31, 14.25, 11.62, 9.14, 7.26,
        5.83, 4.61, 3.72, 3.05, 2.54
    ],
    "val_loss": [
        30.15, 29.14, 27.02, 24.81, 22.63,
        20.45, 18.72, 17.91, 17.25, 16.93,
        17.14, 17.82, 18.46, 19.31, 20.27
    ],

    "train_acc": [
        0.11, 0.21, 0.32, 0.43, 0.54,
        0.63, 0.71, 0.78, 0.84, 0.89,
        0.92, 0.95, 0.97, 0.98, 0.99
    ],
    "val_acc": [
        0.16, 0.26, 0.35, 0.43, 0.50,
        0.56, 0.61, 0.65, 0.68, 0.70,
        0.72, 0.73, 0.71, 0.70, 0.68
    ],

    "train_precision": [
        0.08, 0.14, 0.24, 0.35, 0.47,
        0.58, 0.67, 0.75, 0.82, 0.87,
        0.91, 0.94, 0.96, 0.98, 0.99
    ],
    "val_precision": [
        0.11, 0.16, 0.25, 0.35, 0.44,
        0.51, 0.57, 0.62, 0.66, 0.69,
        0.71, 0.72, 0.70, 0.68, 0.67
    ],

    "train_recall": [
        0.08, 0.16, 0.27, 0.38, 0.49,
        0.59, 0.68, 0.76, 0.83, 0.88,
        0.92, 0.95, 0.97, 0.98, 0.99
    ],
    "val_recall": [
        0.11, 0.16, 0.26, 0.36, 0.45,
        0.52, 0.58, 0.63, 0.67, 0.70,
        0.72, 0.71, 0.70, 0.68, 0.66
    ],

    "train_f1": [
        0.08, 0.15, 0.25, 0.36, 0.48,
        0.58, 0.67, 0.75, 0.82, 0.87,
        0.91, 0.94, 0.96, 0.98, 0.99
    ],
    "val_f1": [
        0.08, 0.13, 0.23, 0.34, 0.44,
        0.51, 0.57, 0.62, 0.66, 0.69,
        0.71, 0.715, 0.70, 0.68, 0.66
    ],

    "eer": [
        0.43, 0.39, 0.35, 0.31, 0.27,
        0.24, 0.21, 0.18, 0.16, 0.145,
        0.132, 0.125, 0.129, 0.137, 0.148
    ],

    "far_at_target": [
        0.0098, 0.0097, 0.0095, 0.0094, 0.0091,
        0.0089, 0.0087, 0.0085, 0.0083, 0.0082,
        0.0080, 0.0079, 0.0081, 0.0084, 0.0086
    ],

    "tar_at_far": [
        0.04, 0.08, 0.14, 0.22, 0.31,
        0.41, 0.51, 0.60, 0.68, 0.74,
        0.79, 0.82, 0.81, 0.79, 0.77
    ],

    "auc": [
        0.62, 0.66, 0.70, 0.74, 0.78,
        0.81, 0.84, 0.87, 0.89, 0.91,
        0.925, 0.934, 0.929, 0.921, 0.913
    ]
}

    plot_history(history_test)
