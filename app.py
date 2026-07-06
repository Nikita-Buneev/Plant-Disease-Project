import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime

from src.inference import load_best_model, predict_image
from src.data_loader import get_dataloaders

import portalocker
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'uploads')
app.config['HISTORY_FILE'] = os.path.join(PROJECT_ROOT, 'web_app', 'history.json')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = os.path.join(PROJECT_ROOT, 'data', 'PlantVillage')
_, _, _, class_names = get_dataloaders(data_dir, batch_size=1)

model = load_best_model(device=device, model_dir=os.path.join(PROJECT_ROOT, 'models'))

def save_history(entry):
    print("save_history вызвана")
    history_path = app.config['HISTORY_FILE']
    print(f"Путь к файлу: {history_path}")
    history = []
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append(entry)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"✅ Сохранено {len(history)} записей")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    results = predict_image(model, filepath, class_names, device)
    entry = {
        'timestamp': datetime.now().isoformat(),
        'image': '/static/uploads/' + filename,
        'predictions': results,
        'filename': filename
    }
    save_history(entry)
    return jsonify({'predictions': results})

@app.route('/history')
def history():
    history = []
    if os.path.exists(app.config['HISTORY_FILE']):
        with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
            history = json.load(f)
    return render_template('history.html', history=history)

@app.route('/generate_report')
def generate_report():
    import tempfile
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from flask import send_file

    try:
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        pdfmetrics.registerFont(TTFont('CyrillicFont', font_path))
        font_name = 'CyrillicFont'
    except:
        font_name = 'Helvetica'

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName=font_name, fontSize=18, alignment=TA_CENTER, spaceAfter=20)
    header_style = ParagraphStyle('Header', parent=styles['Normal'], fontName=font_name, fontSize=11, alignment=TA_CENTER, textColor=colors.whitesmoke)
    cell_style = ParagraphStyle('Cell', parent=styles['Normal'], fontName=font_name, fontSize=10, alignment=TA_CENTER)

    fd, pdf_path = tempfile.mkstemp(suffix='.pdf', prefix='report_')
    os.close(fd)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    story = []
    story.append(Paragraph("Сравнение архитектур нейронных сетей", title_style))
    story.append(Spacer(1, 10))

    metrics_path = os.path.join(PROJECT_ROOT, 'runs', 'evaluation_metrics.json')
    if not os.path.exists(metrics_path):
        story.append(Paragraph("Файл с метриками не найден. Запустите evaluate.py.", cell_style))
    else:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        table_data = [
            [Paragraph("Модель", header_style), Paragraph("Accuracy", header_style),
             Paragraph("Precision", header_style), Paragraph("Recall", header_style),
             Paragraph("F1-score", header_style), Paragraph("Время (мс)", header_style),
             Paragraph("Размер (МБ)", header_style)]
        ]
        for name, vals in metrics.items():
            table_data.append([
                Paragraph(name, cell_style),
                Paragraph(f"{vals['accuracy']:.4f}", cell_style),
                Paragraph(f"{vals['precision']:.4f}", cell_style),
                Paragraph(f"{vals['recall']:.4f}", cell_style),
                Paragraph(f"{vals['f1']:.4f}", cell_style),
                Paragraph(f"{vals.get('inference_time_ms', 0):.1f}", cell_style),
                Paragraph(f"{vals.get('model_size_mb', 0):.1f}", cell_style)
            ])

        t = Table(table_data, colWidths=[80,70,70,70,70,70,70])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), font_name),
            ('FONTSIZE', (0,0), (-1,0), 11),
            ('BOTTOMPADDING', (0,0), (-1,0), 10),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('FONTNAME', (0,1), (-1,-1), font_name),
            ('FONTSIZE', (0,1), (-1,-1), 9),
        ]))
        story.append(t)

        best = max(metrics.items(), key=lambda x: x[1]['accuracy'])
        story.append(Spacer(1,15))
        story.append(Paragraph(
            f"<b>Лучшая модель:</b> {best[0]} (Accuracy: {best[1]['accuracy']:.4f}, "
            f"время: {best[1].get('inference_time_ms',0):.1f} мс, размер: {best[1].get('model_size_mb',0):.1f} МБ)",
            cell_style
        ))

    doc.build(story)

    response = send_file(pdf_path, as_attachment=True, download_name='report.pdf')
    @response.call_on_close
    def cleanup():
        try: os.remove(pdf_path)
        except: pass
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)