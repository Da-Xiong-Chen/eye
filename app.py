from flask import Flask, request, render_template, send_file, jsonify
import os
import uuid
from pathlib import Path
from PIL import Image
from waitress import serve

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 模型載入（保護起來避免整個 app crash）
model = None
try:
    print("🔍 嘗試載入 YOLOv5 模型...")
    from yolov5 import YOLOv5
    model = YOLOv5("best.pt", device="cpu")
    print("✅ 模型載入成功")
except Exception as e:
    print("❌ 模型載入失敗：", e)

@app.route('/')
def index():
    return "✅ Flask 已啟動，模型狀態：" + ("已載入" if model else "載入失敗")

@app.route('/analyze', methods=['POST'])
def analyze():
    print("📩 收到 /analyze 請求")
    if model is None:
        return jsonify({"error": "Model failed to load."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file).convert("RGB")
        unique_filename = f"{uuid.uuid4()}.jpg"
        upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        image.save(upload_path)
        print("✅ 圖片已儲存：", upload_path)
    except Exception as e:
        print("❌ 圖片處理失敗：", e)
        return jsonify({"error": f"Image error: {e}"}), 500

    try:
        results = model.predict(upload_path)
        output_dir = os.path.join(RESULT_FOLDER, 'result')
        os.makedirs(output_dir, exist_ok=True)
        results.save(save_dir=output_dir)
        result_path = os.path.join(output_dir, unique_filename)
        print("✅ 推論完成：", result_path)
        return send_file(result_path, mimetype='image/jpeg')
    except Exception as e:
        print("❌ 推論失敗：", e)
        return jsonify({"error": f"Inference error: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 啟動 Flask 於 0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port)




# from flask import Flask, request, jsonify, render_template, send_file
# import os
# from pathlib import Path
# from yolov5 import detect  # YOLOv5 detect 模組
# from werkzeug.utils import secure_filename

# import torch

# # 覆蓋 torch.load 以確保路徑為字符串格式
# original_torch_load = torch.load
# def patched_torch_load(weights, *args, **kwargs):
#     if isinstance(weights, (Path, os.PathLike)):
#         weights = str(weights)
#     return original_torch_load(weights, *args, **kwargs)

# torch.load = patched_torch_load


# app = Flask(__name__)

# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     # 渲染前端 HTML
#     return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     # 檢查是否有文件上傳
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     # 保存文件
#     filename = secure_filename(file.filename)
#     filepath = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
#     file.save(filepath)

#     # 使用 YOLO 模型進行推論
#     weights_path = str(Path('yolov5/best.pt').resolve())  # 確保權重路徑為字符串格式
#     source_path = str(Path(filepath).resolve())          # 上傳文件的絕對路徑

#     results = detect.run(
#         weights=weights_path,
#         source=source_path,
#         save_txt=False,
#         save_conf=False,
#         project=RESULT_FOLDER,
#         name='result',
#         exist_ok=True
#     )

#     # 推論結果圖片路徑
#     result_image_path = os.path.abspath(os.path.join(RESULT_FOLDER, 'result', filename))

#     # 返回圖片給前端
#     return send_file(result_image_path, mimetype='image/png')

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
