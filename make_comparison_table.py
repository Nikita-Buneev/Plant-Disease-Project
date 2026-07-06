import json
import pandas as pd

with open('runs/evaluation_metrics.json', 'r') as f:
    data = json.load(f)

rows = []
for model, metrics in data.items():
    rows.append({
        'Модель': model,
        'Accuracy': round(metrics['accuracy'], 4),
        'Precision (macro)': round(metrics['precision'], 4),
        'Recall (macro)': round(metrics['recall'], 4),
        'F1-score (macro)': round(metrics['f1'], 4),
        'Время инференса (мс)': round(metrics['inference_time_ms'], 2),
        'Размер модели (МБ)': round(metrics['model_size_mb'], 2)
    })

df = pd.DataFrame(rows)
df = df.sort_values('Accuracy', ascending=False)

print('\n=== СРАВНИТЕЛЬНАЯ ТАБЛИЦА АРХИТЕКТУР ===\n')
print(df.to_string(index=False))

df.to_csv('runs/comparison_table.csv', index=False, encoding='utf-8-sig')
print('\n Таблица сохранена в runs/comparison_table.csv')