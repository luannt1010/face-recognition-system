import torch
import argparse
from torch.utils.data import DataLoader
from src.loss_fn import AdaFaceLoss
from src.net import ResNetEncoder
from src.face_dataset import FaceDataset, SubsetFaceDataset
from src.helper import train, define_transform, create_data_splits, plot_history


def get_args():
    parser = argparse.ArgumentParser(description="Train AdaFace ResNet Encoder")
    # Paths
    parser.add_argument("--root_dir", type=str, default=r".\embedded_model\datasets\webface\webface_112x112")
    parser.add_argument("--save_path", type=str, default=r".\runs\checkpoints_adaface")
    # Dataset
    parser.add_argument("--val_factor", type=float, default=0.3)
    # DataLoader
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    # Model
    parser.add_argument("--model_size", type=int, default=18)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    # Loss
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--t_alpha", type=float, default=0.01)
    # Training
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is training on {device}.")

    root_dir = args.root_dir
    dataset = FaceDataset(root_dir=root_dir)
    train_transform, val_transform = define_transform()
    train_dataset, val_dataset = create_data_splits(dataset, val_factor=args.val_factor)
    train_dataset = SubsetFaceDataset(train_dataset, train_transform)
    val_dataset = SubsetFaceDataset(val_dataset, val_transform)
    print(f" Size of train dataset: {len(train_dataset)}")
    print(f" Size of val dataset: {len(val_dataset)}")
    print(f"Total classes: {len(dataset.classes)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            drop_last=False, num_workers=args.num_workers)
    print("Create dataloader successfully!")

    model = ResNetEncoder(model_size=args.model_size, embedding_dim=args.embedding_dim, dropout=args.dropout_rate)
    criterion = AdaFaceLoss(num_classes=len(dataset.classes), embedding_dim=args.embedding_dim,
                            m=args.margin, s=args.scale, t_alpha=args.t_alpha)
    optimizer = torch.optim.AdamW(params=list(model.parameters()) + list(criterion.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min")
    history = train(model, train_loader, val_loader, args.num_epochs, optimizer, criterion,
                    save_path=args.save_path, scheduler=scheduler, device=device)
    plot_history(history)


if __name__ == "__main__":
    main()

