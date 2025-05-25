import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, multilabel_confusion_matrix
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Zaman ölçümünü başlat
start_time = time.perf_counter()

# === 1. Veri kümesini yükle ===
df = pd.read_csv("bbc-text.csv")
df = df[['category', 'text']].dropna()

# Etiketleri sayısal değerlere dönüştür
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
y = df['label']


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])


models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Linear SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier(),
    'kNN': KNeighborsClassifier(n_neighbors=3),
    'MLP': MLPClassifier(max_iter=500),
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# === 4. Performans Ölçümleri ve Çapraz Doğrulama ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
performance_metrics = []
conf_matrices = {}

for name, model in models.items():
    print(f"Model: {name}")
    y_pred = cross_val_predict(model, X, y, cv=cv)

    cm = confusion_matrix(y, y_pred)
    conf_matrices[name] = cm

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro', zero_division=0)
    recall = recall_score(y, y_pred, average='macro', zero_division=0)
    specificity = []
    mcm = multilabel_confusion_matrix(y, y_pred, labels=np.unique(y))
    for i in range(len(mcm)):
        tn, fp, fn, tp = mcm[i].ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    specificity = np.mean(specificity)

    f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)

    performance_metrics.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'MCC': mcc
    })

# === 5. Performans Matrisi ===
performance_df = pd.DataFrame(performance_metrics)
performance_df = performance_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 4))
sns.set(font_scale=1)
sns.heatmap(performance_df.drop(columns='Model').set_index(performance_df['Model']).round(3),
            annot=True, cmap="YlGnBu", fmt=".3f", cbar=True)
plt.title("Sınıflandırma Modelleri Performans Matrisi", fontsize=14)
plt.xlabel("Metrikler")
plt.ylabel("Modeller")
plt.tight_layout()
plt.show()

# === 6. Karmaşıklık Matrisleri ===
for name, cm in conf_matrices.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"Karmaşıklık Matrisi - {name}")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.show()

# === Sınıf Bazlı Performans Görseli ===
y_pred = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv)
report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
class_metrics = report_df.iloc[:-3][['precision', 'recall', 'f1-score']]

plt.figure(figsize=(10, 6))
sns.heatmap(class_metrics, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Sınıf Bazlı Precision / Recall / F1 Skorları')
plt.xlabel('Metrik')
plt.ylabel('Kategori')
plt.tight_layout()
plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nToplam Çalışma Süresi: {elapsed_time:.2f} saniye")
