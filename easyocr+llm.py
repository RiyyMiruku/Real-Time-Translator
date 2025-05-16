import time
import mss
import numpy as np
import cv2
import easyocr
import tkinter as tk
from transformers import MarianTokenizer, MarianMTModel

# 初始化 OCR 與翻譯模型
reader = easyocr.Reader(['en', 'ch_sim'])
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 翻譯函式
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 簡單圖片預處理
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
  
    return enhanced

# UI 初始化
root = tk.Tk()
label = tk.Label(root, text="⏳ 啟動中...", font=("Arial", 16), bg="white")
label.pack()
root.attributes("-topmost", True)
root.overrideredirect(True)

# 螢幕截圖範圍 (手動調整)
bbox = {'top': 200, 'left': 500, 'width': 400, 'height': 100}

# 主迴圈
def main_loop():
    prev_text = ""
    with mss.mss() as sct:
        while True:
            raw_img = np.array(sct.grab(bbox))
            proc_img = preprocess_image(raw_img)
            cv2.imshow("Preview", proc_img)
            cv2.waitKey(1)
            result = reader.readtext(proc_img, detail=0)
            text = " ".join(result).strip()
            print(text)

            if text and text != prev_text:
                translated = translate_text(text)
                label.config(text=translated)
                prev_text = text

            root.geometry(f"+{bbox['left']}+{bbox['top'] + bbox['height'] + 10}")
            root.update()
            time.sleep(0.2)

# 執行主程式
main_loop()
