import time
import mss
import numpy as np
import cv2
import easyocr
import argostranslate.package
import argostranslate.translate
import tkinter as tk
import threading
import keyboard
import pyautogui
from urllib.request import urlopen
# ========== 區域選擇器 ==========
def select_bbox():
    print("請用滑鼠點選左上角與右下角")
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"點擊: {x}, {y}")
            if len(points) == 2:
                cv2.destroyAllWindows()

    screen = pyautogui.screenshot()
    screen_np = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
    clone = screen_np.copy()

    cv2.namedWindow("選取區域")
    cv2.setMouseCallback("選取區域", mouse_callback)

    while len(points) < 2:
        disp = clone.copy()
        if len(points) == 1:
            cv2.rectangle(disp, points[0], pyautogui.position(), (0, 255, 0), 2)
        cv2.imshow("選取區域", disp)
        if cv2.waitKey(10) == 27:  # ESC 離開
            break

    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        return {
            "left": min(x1, x2),
            "top": min(y1, y2),
            "width": abs(x2 - x1),
            "height": abs(y2 - y1)
        }
    else:
        return None

# ========== 初始化 ==========
bbox = select_bbox()
if not bbox:
    print("未選擇區域，程式結束")
    exit()

prev_img = None
prev_text = ""
paused = False

reader = easyocr.Reader(['en'], gpu=False)
# argostranslate.translate.load_installed_languages()

def ensure_translation_installed(from_code="en", to_code="zh"):
    # 載入目前安裝的語言
    argostranslate.translate.load_installed_languages()
    try:
        # 嘗試取得翻譯器
        translation = argostranslate.translate.get_translation_from_codes(from_code, to_code)
        return translation
    
    except Exception:
        print(f"⚠️ 尚未安裝 {from_code} → {to_code}，正在下載...")
        # 嘗試下載語言包
        try:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                filter(
                    lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
                )
            )
            argostranslate.package.install_from_path(package_to_install.download())
            print(f"✅ 成功安裝 {from_code} → {to_code} 語言包")
            return translation
        
        except Exception as e:
            print("❌ 安裝語言包時發生錯誤：", e)
            return None

ensure_translation_installed("en", "zh")
# ========== GUI ==========
root = tk.Tk()
root.attributes("-topmost", True)
root.overrideredirect(True)
label = tk.Label(root, text="", font=("Arial", 16), bg="black", fg="white", wraplength=400)
label.pack()

# ========== 功能 ==========
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)
    return contrast

def get_changed_regions(current_img):
    global prev_img
    if prev_img is None:
        prev_img = current_img
        return [current_img]

    diff = cv2.absdiff(current_img, prev_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            regions.append(current_img[y:y+h, x:x+w])

    prev_img = current_img
    return regions

def translate_text(text):
    from_code, to_code = "en", "zh"
    translation = argostranslate.translate.get_translation_from_codes(from_code, to_code)
    return translation.translate(text)
# ========== 暫停功能 ==========
def toggle_pause():
    global paused
    paused = not paused
    print("🔴 暫停中" if paused else "🟢 已恢復")

keyboard.add_hotkey('F8', toggle_pause)

def main_loop():
    global prev_text
    with mss.mss() as sct:
        while True:
            if paused:
                label.config(text="🔴 暫停中 (F8 切換)")
                root.geometry(f"+{bbox['left']}+{bbox['top'] + bbox['height'] + 10}")
                root.update()
                time.sleep(0.3)
                continue

            # 擷取畫面
            raw_img = np.array(sct.grab(bbox))

            # 預處理（轉灰階＋提高對比）
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

            # OCR 辨識
            result = reader.readtext(contrast, detail=0)

            # 組成一行文字
            text = " ".join(result).strip()

            # 翻譯與更新顯示
            if text and text != prev_text:
                translated = translate_text(text)
                label.config(text=translated)
                root.geometry(f"+{bbox['left']}+{bbox['top'] + bbox['height'] + 10}")
                prev_text = text

            root.update()
            time.sleep(0.3)

# ========== 啟動主執行緒 ==========
threading.Thread(target=main_loop, daemon=True).start()
root.mainloop()
