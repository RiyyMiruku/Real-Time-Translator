import time
import mss
import numpy as np
import cv2
import pytesseract
import tkinter as tk
from pynput import keyboard
from threading import Thread
from transformers import MarianTokenizer, MarianMTModel
from PIL import Image, ImageDraw, ImageFont

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

# 翻譯函式 (此處假設僅顯示辨識文字)
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)



def draw_text(img_bgr, text):
    # 將 OpenCV 的 BGR 圖片轉為 PIL 的 RGB 圖片
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 建立繪圖對象
    draw = ImageDraw.Draw(pil_img)

    font_path = "C:/Windows/Fonts/msjh.ttc"
    font = ImageFont.truetype(font_path, 24)

    # 畫上文字
    draw.text((10, 10), text, font=font, fill=(0, 255, 0))

    # 轉回 OpenCV 圖片格式
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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

# 主程式邏輯
def main_loop():
    global bbox, translated_text
    prev_text = ""
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])

    with mss.mss() as sct:
        while running:
            if paused or bbox is None:
                time.sleep(0.1)
                continue

            screenshot = np.array(sct.grab(bbox))
            sharp = cv2.filter2D(screenshot, -1, sharpen_kernel)
            text = pytesseract.image_to_string(sharp)
            text = text.strip()

            if text and text != prev_text:
                translated_text = translate_text(text)
                prev_text = text

            # 顯示畫面與翻譯
            display_img = draw_text(sharp.copy(),translated_text)
            img_pil = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            # img_pil = cv2.putText(img_pil, translated_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Translation Viewer", img_pil)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.7)

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
