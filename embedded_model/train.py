import torch
import argparse
from torch.utils.data import DataLoader
from src.loss_fn import AdaFaceLoss, ArcFaceLoss, BatchHardTripletLoss
from src.face_dataset import FaceDataset, SubsetFaceDataset
from src.helper import train, define_transform, create_data_splits, plot_history, load_model


def get_args():
    parser = argparse.ArgumentParser(description="Train AdaFace ResNet Encoder")
    # Paths
    parser.add_argument("--root_dir", type=str, default=r".\datasets\webface\webface_112x112")
    parser.add_argument("--save_path", type=str, default=r".\runs\checkpoints_adaface")
    # Dataset
    parser.add_argument("--val_factor", type=float, default=0.3)
    # DataLoader
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    # Model
    parser.add_argument("--model_type", type=str, default="iresnet")
    parser.add_argument("--model_size", type=int, default=18)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    # Loss
    parser.add_argument("--loss_type", type=str, default="arc")
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
    root_dir = args.root_dir
    save_path = args.save_path
    val_factor = args.val_factor
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_type = args.model_type
    model_size = args.model_size
    embedding_dim = args.embedding_dim
    dropout_rate = args.dropout_rate
    loss_type = args.loss_type
    margin = args.margin
    scale = args.scale
    t_alpha = args.t_alpha
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{model_type.title()} model is training on {device} with {loss_type.title()} Loss.")

    dataset = FaceDataset(root_dir=root_dir)
    train_transform, val_transform = define_transform()
    train_dataset, val_dataset = create_data_splits(dataset, val_factor=val_factor)
    train_dataset = SubsetFaceDataset(train_dataset, train_transform)
    val_dataset = SubsetFaceDataset(val_dataset, val_transform)
    print(f" Size of train dataset: {len(train_dataset)}")
    print(f" Size of val dataset: {len(val_dataset)}")
    print(f" Total classes: {len(dataset.classes)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(),
                              drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(),
                            drop_last=False, num_workers=num_workers)
    print("Create dataloader successfully!")

    model = load_model(model_type, model_size, embedding_dim, dropout_rate, None)

    if loss_type == "ada":
        criterion = AdaFaceLoss(num_classes=len(dataset.classes), embedding_dim=args.embedding_dim, m=args.margin, s=scale, t_alpha=t_alpha)
    elif loss_type == "arc":
        criterion = ArcFaceLoss(num_classes=len(dataset.classes), embedding_dim=args.embedding_dim, m=args.margin, s=scale)
    elif loss_type == "triplet":
        criterion = BatchHardTripletLoss(margin=margin)
    optimizer = torch.optim.AdamW(params=list(model.parameters()) + list(criterion.parameters()),
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min")
    history = train(model, train_loader, val_loader, num_epochs, optimizer, criterion,
                    save_path=save_path, scheduler=scheduler, device=device)
    plot_history(history, save_path)


if __name__ == "__main__":
    main()

