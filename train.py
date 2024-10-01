import logging
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.adamax import Adamax
from dataset import dataset
from model.stegcl import stegcl_loss
from opts.options import arguments
from model.model import Srnet
from utils.utils import (
    latest_checkpoint,
    adjust_learning_rate,
    weights_init,
    saver,
)
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params
opt = arguments()

logging.basicConfig(
    filename="training.log",
    format="%(asctime)s %(message)s",
    level=logging.DEBUG,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



if __name__ == "__main__":

    train_data = dataset.DatasetLoad(
        opt.cover_path,
        opt.stego_path,

        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        ),
    )

    val_data = dataset.DatasetLoad(
        opt.valid_cover_path,
        opt.valid_stego_path,

        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=True,num_workers=16 )
    valid_loader = DataLoader(
        val_data, batch_size=opt.batch_size, shuffle=False,num_workers=16
    )

    model = Srnet()
    model.to(device)
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-Trainable Parameters: {non_trainable_params}")

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = Adamax(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=1e-4)

    check_point = latest_checkpoint()
    if not check_point:
        START_EPOCH = 1
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)
        print("No checkpoints found!!, Retraining started... ")
    else:
        pth = opt.checkpoints_dir + "net_" + str(check_point) + ".pt"
        ckpt = torch.load(pth)
        START_EPOCH = ckpt["epoch"] + 1
        pretrained_dict = ckpt["model_state_dict"]
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and model_dict[k].size() == v.size()}

        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)


        print("Model Loaded from epoch " + str(START_EPOCH) + "..")

    for epoch in range(START_EPOCH, opt.num_epochs + 1):
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        test_accuracy = []
        total_covers = 0
        total_stegos = 0
        correctly_classified_covers = 0
        correctly_classified_stegos = 0
        val_total_covers=0
        val_total_stegos=0
        val_correctly_classified_covers=0
        val_correctly_classified_stegos=0
        # Training
        model.train()
        st_time = time.time()
        adjust_learning_rate(optimizer, epoch)

        for i, train_batch in enumerate(train_loader):
            images = torch.cat((train_batch["cover"], train_batch["stego"]), 0)
            labels = torch.cat((train_batch["label"][0], train_batch["label"][1]), 0)

            # 创建一个随机排列的索引
            shuffled_indices = torch.randperm(images.size(0))

            # 使用索引来打乱图像和标签
            shuffled_images = images[shuffled_indices]
            shuffled_labels = labels[shuffled_indices]

            # 将打乱顺序后的图像和标签送入模型
            shuffled_images = shuffled_images.to(device, dtype=torch.float)
            shuffled_labels = shuffled_labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(shuffled_images)
            loss = loss_fn(outputs, shuffled_labels)
            loss.backward()

            optimizer.step()
            training_loss.append(loss.item())
            prediction = outputs.data.max(1)[1]
            accuracy = (
                prediction.eq(shuffled_labels.data).sum() * 100.0 / (shuffled_labels.size()[0])
            )
            training_accuracy.append(accuracy.item())
            total_covers += (shuffled_labels == 0).sum().item()
            total_stegos += (shuffled_labels == 1).sum().item()
            correctly_classified_covers += ((prediction == 0) & (shuffled_labels == 0)).sum().item()
            correctly_classified_stegos += ((prediction == 1) & (shuffled_labels == 1)).sum().item()

            # 计算训练的错误率

            sys.stdout.write(
                f"\r Epoch:{epoch}/{opt.num_epochs}"
                f" Batch:{i+1}/{len(train_loader)}"
                f" Loss:{training_loss[-1]:.4f}"
                f" Acc:{training_accuracy[-1]:.2f}"
                f" LR:{optimizer.param_groups[0]['lr']:.4f}"
            )
        train_err = 1 - (correctly_classified_covers + correctly_classified_stegos) / (total_covers + total_stegos)
        end_time = time.time()

        model.eval()
        total = 0
        correct = 0
        validation_loss = []
        with torch.no_grad():
            for i, val_batch in enumerate(valid_loader):
                images = torch.cat((val_batch["cover"], val_batch["stego"]), 0)
                labels = torch.cat((val_batch["label"][0], val_batch["label"][1]), 0)

                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                validation_loss.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_total_covers += (labels == 0).sum().item()
                val_total_stegos += (labels == 1).sum().item()
                val_correctly_classified_covers += ((predicted == 0) & (labels == 0)).sum().item()
                val_correctly_classified_stegos += ((predicted == 1) & (labels == 1)).sum().item()

        val_err = 1 - (val_correctly_classified_covers + val_correctly_classified_stegos) / (val_total_covers + val_total_stegos)
        fpr = (val_total_covers - val_correctly_classified_covers) / val_total_covers
        fnr = (val_total_stegos - val_correctly_classified_stegos) / val_total_stegos
        accuracy = 100.0 * correct / total
        validation_accuracy.append(accuracy)

        avg_train_loss = sum(training_loss) / len(training_loss)
        avg_valid_loss = sum(validation_loss) / len(validation_loss)

        message = (
            f"Epoch: {epoch}. "
            f"Train Loss:{(sum(training_loss) / len(training_loss)):.5f}. "
            f"Valid Loss:{(sum(validation_loss) / len(validation_loss)):.5f}. "
            "Train"
            f" Acc:{(sum(training_accuracy) / len(training_accuracy)):.5f} "
            "Valid"
            f" Acc:{(sum(validation_accuracy) / len(validation_accuracy)):.5f} "
            "fnr"
            f" fnr: {fnr:.5f}"
            "fpr"
            f" fpr: {fpr:.5f}"
        )
        print("\n", message)

        logging.info(message)

        state = {
            "epoch": epoch,
            "opt": opt,
            "train_loss": sum(training_loss) / len(training_loss),
            "valid_loss": sum(validation_loss) / len(validation_loss),
            "train_accuracy": sum(training_accuracy) / len(training_accuracy),
            "valid_accuracy": sum(validation_accuracy) / len(validation_accuracy),
            "train_err": train_err,
            "valid_err": val_err,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
        }

        saver(state, opt.checkpoints_dir, epoch)
