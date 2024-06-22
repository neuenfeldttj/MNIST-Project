import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision import transforms
from mnist import MNISTModel
import os
import io

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Prediction")

        self.canvas = tk.Canvas(root, bg="black", width=500, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.pen_color = "white"
        self.drawing = False
        self.last_x, self.last_y = None, None

        self.image = Image.new('L', (500, 500), 'black')  # Initialize our own image for drawing
        self.draw_img = ImageDraw.Draw(self.image)

        self.lines = []

        self.erase_button = tk.Button(root, text="Erase", command=self.erase)
        self.erase_button.pack(side=tk.RIGHT)

        self.pred_button = tk.Button(root, text="Predict", command=self.predict)
        self.pred_button.pack(side=tk.LEFT)

        self.textbox = tk.Entry(root, width=20)
        self.textbox.pack()

        self.canvas.bind("<Button-1>", self.start_draw) #trackpad
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Load PyTorch model
        try:
            filename = os.path.join("./final_model.pt")
            checkpoint = torch.load(filename)
            self.model = MNISTModel()
            self.model.load_state_dict(checkpoint["state_dict"])
        except Exception:
            print("Could not load file")

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill=self.pen_color, width=8)
            self.draw_img.line((self.last_x, self.last_y, x, y), fill='white', width=8)  # Draw on our image
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.drawing = False

    def erase(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (500, 500), 'black')  # Reset our own image for drawing
        self.draw_img = ImageDraw.Draw(self.image)

    def predict(self):
        img = self.image.resize((28,28))
        img = img.point(lambda p: p > 10 and 255)
        image_array = np.array(img)

        image_tensor = transforms.ToTensor()(image_array)

        image_tensor = image_tensor.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(image_tensor)
            prediction = torch.argmax(prediction, dim=1).item() # takes the index of max probability

        self.textbox.delete(0, tk.END)
        self.textbox.insert(0, str(prediction))


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()