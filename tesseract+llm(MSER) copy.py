import time
import mss
import numpy as np
import cv2
import pytesseract
import tkinter as tk
from pynput import keyboard
from threading import Thread
from transformers import MarianTokenizer, MarianMTModel
from PIL import Image, ImageTk, ImageFont, ImageDraw

# 初始化 Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# 初始化 翻譯模型
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
# 變數控制
paused = False
running = True
bbox = None  # 選取區域將儲存在這
translated_text = ""
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# 翻譯函式

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 熱鍵監聽函式

def on_press(key):
    global paused, running
    try:
        if key == keyboard.Key.f8:
            paused = not paused
        elif key == keyboard.Key.esc:
            running = False
            return False  # 停止 listener
    except:
        pass

# 選取區域 GUI
class ScreenSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.start_x = self.start_y = 0
        self.rect = None
        self.canvas = tk.Canvas(self.root, cursor="cross", bg='gray11')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.attributes("-alpha", 0.3)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<ButtonPress-1>", self.on_press)
        self.root.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<ButtonRelease-1>", self.on_release)
        self.root.mainloop()

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        global bbox
        self.root.destroy()
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y
        bbox = {
            "left": min(x1, x2),
            "top": min(y1, y2),
            "width": abs(x2 - x1),
            "height": abs(y2 - y1)
        }

# MSER 初始化
mser = cv2.MSER_create()

def filter_text_regions(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = cv2.filter2D(l, -1, sharpen_kernel)

    regions, _ = mser.detectRegions(l)

    mask = np.zeros_like(l)
    for p in regions:
        x, y, w, h = cv2.boundingRect(p)
        aspect_ratio = w / float(h)
        if 0.3 < aspect_ratio < 5:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    cv2.imshow("Original sharp", l)
    cv2.imshow("MSER Mask", mask)
    cv2.imshow("Masked result",cv2.bitwise_and(image, image, mask=mask) )

    return cv2.bitwise_and(image, image, mask=mask)

# 顯示 PIL 畫面含中文

def display_with_text(image_np, text):
    image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text((10, 10), text, font=font, fill=(0, 255, 0))
    return np.array(image)

# 主程式邏輯

def main_loop():
    global bbox, translated_text
    prev_text = ""

    with mss.mss() as sct:
        while running:
            if paused or bbox is None:
                time.sleep(0.1)
                continue

            screenshot = np.array(sct.grab(bbox))
            filtered = filter_text_regions(screenshot)
          
            text = pytesseract.image_to_string(filtered, lang='eng')
            text = text.strip()

            if text and text != prev_text:
                translated_text = translate_text(text)
                prev_text = text

            # 顯示畫面與翻譯
            display_img = display_with_text(screenshot, translated_text)
            cv2.imshow("Translation Viewer", display_img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.3)

    cv2.destroyAllWindows()

# 啟動程式
if __name__ == "__main__":
    print("請使用滑鼠拉選螢幕區域...")
    ScreenSelector()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    thread = Thread(target=main_loop)
    thread.start()

    thread.join()
    print("程式結束。")
