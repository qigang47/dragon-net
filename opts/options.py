import argparse


def arguments() -> str:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cover_path",
        default=r"D:\网络空间安全编程\期末大作业\DragenNet\BOSSbase256\cover",
    )
    parser.add_argument(
        "--stego_path",
        default=r"D:\网络空间安全编程\期末大作业\DragenNet\BOSSbase256\stego1bpp",
    )
    parser.add_argument(
        "--valid_cover_path",
        default=r"D:\网络空间安全编程\期末大作业\DragenNet\BOSSbase256\cover_valid",
    )
    parser.add_argument(
        "--valid_stego_path",
        default=(
            r"D:\网络空间安全编程\期末大作业\DragenNet\BOSSbase256\stego1bpp_valid"
        ),
    )
    parser.add_argument("--checkpoints_dir", default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imageSize", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=278)
    parser.add_argument("--lr", type=float, default=0.001)

    opt = parser.parse_args()
    return opt