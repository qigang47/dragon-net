from glob import glob
import torch
import numpy as np
import imageio as io
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from model.model import Srnet

COVER_PATH = None
CHKPT = r"D:\网络空间安全编程\期末大作业\DragenNet\checkpoints\net_60.pt"

def select_cover_path(root):
    global COVER_PATH
    COVER_PATH = filedialog.askdirectory()
    root.destroy()

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    window.geometry(f"{width}x{height}+{x}+{y}")

path_selection_root = tk.Tk()
path_selection_root.title("选择图片")

center_window(path_selection_root, 500, 400)

image_path = r"D:\网络空间安全编程\期末大作业\DragenNet\image.png"
img = Image.open(image_path)
img.thumbnail((400, 300))
photo = ImageTk.PhotoImage(img)

image_label = tk.Label(path_selection_root, image=photo)
image_label.photo = photo
image_label.place(x=50, y=80)

cover_path_button = tk.Button(path_selection_root, text="打开文件夹",
                              bg="#323071", fg="white",
                              font=("Arial Rounded MT Bold", 16),
                              command=lambda: select_cover_path(path_selection_root),
                              relief="groove",
                              bd=2,
                              highlightbackground="#0053A6",
                              activebackground="#7A77BE",
                              activeforeground="white")

cover_path_button.place(x=125, y=220, width=250)

path_selection_root.mainloop()

if COVER_PATH is not None:
    root = tk.Tk()
    root.title("预测")

    center_window(root, 500, 500)

    frame = tk.Frame(root)
    frame.pack()

    cover_image_names = glob(COVER_PATH + '/*.pgm')

    model = Srnet().cuda()

    ckpt = torch.load(CHKPT)
    model.load_state_dict(ckpt["model_state_dict"])
    images = torch.empty((1, 1, 256, 256), dtype=torch.float)
    test_accuracy = []

    current_index = 0


    def update_gui(image, prediction):
        img = Image.open(image)
        img = img.resize((256, 256), Image.ANTIALIAS if 'ANTIALIAS' in dir(Image) else 3)
        img = ImageTk.PhotoImage(img)

        panel = tk.Label(frame, image=img)
        panel.image = img
        panel.grid(row=0, column=0, padx=0, pady=40)

        index_label = tk.Label(frame, text=f"序号：{current_index + 1}/{len(cover_image_names)}")
        index_label.grid(row=1, column=0, padx=10)
        label_type = '是' if prediction == 1 else '否'
        pred_label = tk.Label(frame, text=f"是否隐写: {label_type}")
        pred_label.grid(row=2, column=0, padx=10, pady=10)


    def next_image():
        global current_index

        if current_index < len(cover_image_names):
            image_path = cover_image_names[current_index]
            images = [image_path]
            image_tensor = torch.empty((1, 1, 256, 256), dtype=torch.float).cuda()
            image_tensor[0, 0, :, :] = torch.tensor(io.imread(image_path)).cuda()

            outputs = model(image_tensor)
            prediction = outputs.data.max(1)[1].item()

            update_gui(image_path, prediction)

            current_index += 1

            if current_index == len(cover_image_names):
                next_image_button.config(text="关闭",
                                         command=root.destroy,
                                         bg="#1A12F5",
                                         fg="white",
                                         font=("Arial Rounded MT Bold", 16),
                                         relief="groove",
                                         bd=2,
                                         highlightbackground="#1E1C45",
                                         activebackground="#4E48E9",
                                         activeforeground="white")


    next_image_button = tk.Button(root, text="下一张",
                                   command=next_image,
                                   bg="#1A12F5",
                                   fg="white",
                                   font=("Arial Rounded MT Bold", 16),
                                   relief="groove",
                                   bd=2,
                                   highlightbackground="#1E1C45",
                                   activebackground="#4E48E9",
                                   activeforeground="white")
    next_image_button.pack()

    next_image()

    root.mainloop()
else:
    print("未选择文件夹，程序退出")
