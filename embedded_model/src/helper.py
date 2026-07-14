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
from .metrics import classification_metrics, l2_normalize, verification_metrics_report
from .net import ResNetEncoder

def load_model_frome_state_dict(sd_path, model_size=18, embedding_dim=512, dropout_rate=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetEncoder(model_size, embedding_dim, dropout_rate)
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
    train_transform = transforms.Compose([transforms.Resize((112, 112)),
                                          transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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

    loss_fn = loss_fn.to(device)
    model = model.to(device)
    best_score = float("inf")
    total_time = 0.0
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        loss_fn.train()
        train_running_loss = 0
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
        train_metrics = classification_metrics(train_labels, train_preds)
        train_acc = train_metrics["accuracy"]
        train_precision = train_metrics["precision"]
        train_recall = train_metrics["recall"]
        train_f1 = train_metrics["f1"]

        model.eval()
        loss_fn.eval()
        val_running_loss = 0
        val_preds, val_labels = [], []
        val_embeddings = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validating]", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                embedding = model(images)
                loss = loss_fn(embedding, labels)
                val_running_loss += loss.item()

                w = loss_fn.W.detach().cpu().numpy()
                embedding_np = embedding.detach().cpu().numpy()
                w_norm = l2_normalize(w, epsilon)
                emb_norm = l2_normalize(embedding_np, epsilon)
                preds = np.argmax(np.dot(emb_norm, w_norm.T), axis=1)
                val_preds.extend(preds.tolist())
                val_labels.extend(labels.detach().cpu().tolist())
                val_embeddings.append(embedding.detach().cpu())

        val_epoch_loss = val_running_loss / len(val_loader)
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


def plot_history(history):
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
    val_verification_acc = history.get("val_verification_acc", [])
    val_verification_eer = history.get("val_verification_eer", [])
    val_verification_roc_auc = history.get("val_verification_roc_auc", [])
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
    if val_verification_acc:
        ax[1].plot(epochs, val_verification_acc, label="Val Verification Acc")
    if val_verification_roc_auc:
        ax[1].plot(epochs, val_verification_roc_auc, label="Val Verification ROC-AUC")
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

    # F1
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
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

    if val_verification_eer:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        idx = np.argmin(val_verification_eer)
        best_epoch = epochs[idx]
        best_val = val_verification_eer[idx]
        ax.plot(epochs, val_verification_eer, label="Val Verification EER")
        ax.scatter(best_epoch, best_val, s=50)
        ax.annotate(
            f"Best Val EER\n({best_epoch}, {best_val:.4f})",
            xy=(best_epoch, best_val),
            xytext=(20, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red")
        ax.set_title("Validation Verification EER")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("EER")
        ax.legend()

    plt.tight_layout()
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
        img1_emb = model(img1.unsqueeze(0))
        img2_emb = model(img2.unsqueeze(0))
        if mode == "cosine":
            result = F.cosine_similarity(img1_emb, img2_emb)
        elif mode == "euclid":
            img1_emb = F.normalize(img1_emb, 2, 1)
            img2_emb = F.normalize(img2_emb, 2, 1)
            result = torch.sqrt(torch.sum(torch.pow(img1_emb - img2_emb, 2), 1))
    return result, img1_emb, img2_emb

