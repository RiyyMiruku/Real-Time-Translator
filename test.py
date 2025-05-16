import time
import mss
import numpy as np
import cv2
import pytesseract
import tkinter as tk
from pynput import keyboard
from threading import Thread
from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image, ImageDraw, ImageFont
import ctypes
import torch
import multiprocessing
import re

#監測用
import time
import psutil
import os

class Profiler:
    def __init__(self):
        self.start_time = None
        self.records = []

    def start(self, tag=""):
        self.start_time = time.time()
        self.start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.tag = tag

    def end(self):
        if self.start_time is None:
            print("⚠️ 請先呼叫 start()")
            return
        end_time = time.time()
        end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        elapsed = end_time - self.start_time
        mem_used = end_mem - self.start_mem
        self.records.append((self.tag, elapsed, mem_used))
        print(f"[{self.tag}] 耗時: {elapsed:.4f}s, 記憶體變化: {mem_used:.2f} MB")
        self.start_time = None  # reset for next usage

    def summary(self):
        print("\n📋 Profiler 總結：")
        for tag, t, m in self.records:
            print(f" - [{tag}] 時間: {t:.4f}s, 記憶體: {m:.2f} MB")

#翻譯器           
class Translator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-zh"):
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 使用設備：{self.device}")

        if torch.cuda.is_available():
            print("[INFO] 使用 DeepSeek 翻譯模型 (GPU)...")
            model_name = "deepseek-ai/deepseek-translate"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        else:
            print("[INFO] 使用 Opus MT 翻譯模型 (CPU)...")
            model_name = "Helsinki-NLP/opus-mt-en-zh"
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def _split_text(self, text, max_chars=200):
        # 根據標點分句並控制單段長度
        sentences = re.split(r'(?<=[。！？!?\.])\s*', text.strip())
        segments, buffer = [], ""
        for sentence in sentences:
            if len(buffer) + len(sentence) < max_chars:
                buffer += sentence
            else:
                if buffer:
                    segments.append(buffer)
                buffer = sentence
        if buffer:
            segments.append(buffer)
        return segments

    def parallel_translate(self, text):
        segments = self._split_text(text)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.translate, segments)
        return "\n".join(results)


# 變數控制
paused = False
running = True
bbox = None  # 選取區域將儲存在這
translated_text = ""

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

# 選取區域 GUI/考量dpi縮放比例
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    ctypes.windll.user32.SetProcessDPIAware()

class ScreenSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-alpha", 0.3)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="gray11")
        self.canvas = tk.Canvas(self.root, cursor="cross", bg='gray11')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_x = self.start_y = 0
        self.rect = None

        self.root.bind("<ButtonPress-1>", self.on_press)
        self.root.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<ButtonRelease-1>", self.on_release)
        self.root.mainloop()

    def on_press(self, event):
        self.start_x = self.root.winfo_pointerx()
        self.start_y = self.root.winfo_pointery()
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_drag(self, event):
        current_x = self.root.winfo_pointerx()
        current_y = self.root.winfo_pointery()
        self.canvas.coords(self.rect, self.start_x, self.start_y, current_x, current_y)

    def on_release(self, event):
        global bbox
        end_x = self.root.winfo_pointerx()
        end_y = self.root.winfo_pointery()
        self.root.destroy()
        bbox = {
            "left": min(self.start_x, end_x),
            "top": min(self.start_y, end_y),
            "width": abs(end_x - self.start_x),
            "height": abs(end_y - self.start_y)
        }


# 主程式邏輯
def main_loop():
    global bbox, translated_text, paused, running
    # 初始化 Tesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 
    
    prev_text = ""
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])

    with mss.mss() as sct:
      
        while paused or bbox is None:
            time.sleep(1)
            
        prof.start("Preprocessing")
        screenshot = np.array(sct.grab(bbox))
        sharp = cv2.filter2D(screenshot, -1, sharpen_kernel)
        prof.end()

        prof.start("OCR")
        text = pytesseract.image_to_string(sharp)
        text = text.strip()
        prof.end()

        
        if text and text != prev_text:
            #測試單線程與多線程
            prof.start("Translation")
            translated_text1 = translator.translate(text)
            prof.end()

            prof.start("multitranslation")
            translated_text2 = translator.parallel_translate(text)
            prof.end()

            prev_text = text
            
        print(f"辨識文字: {text}")
        print(f"翻譯文字: {translated_text1}")
        print(f"翻譯文字: {translated_text2}")

        # 顯示畫面與翻譯
        prof.start("Display")
        display_img = draw_text(sharp.copy(),translated_text)
        img_pil = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        # img_pil = cv2.putText(img_pil, translated_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Translation Viewer", img_pil)
        prof.end()


    # cv2.destroyAllWindows()

# 啟動程式
if __name__ == "__main__":
    print("請使用滑鼠拉選螢幕區域...")
    #選取區域 GUI，直到選取結束
    ScreenSelector()
    #啟動紀錄器
    prof = Profiler()
    #啟動翻譯器
    translator = Translator()
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()

    thread = Thread(target=main_loop)
    thread.start()

    thread.join()
    prof.summary()
    print("程式結束。")
