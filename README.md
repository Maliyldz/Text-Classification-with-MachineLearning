# Text-Classification-with-MachineLearning
## 📚 BBC Haber Metin Sınıflandırması

Bu proje, BBC haberlerinden oluşan bir veri kümesi üzerinde çeşitli makine öğrenmesi modelleriyle metin sınıflandırması gerçekleştirilmektedir. Proje kapsamında TF-IDF vektörleştirme yöntemi kullanılmış ve farklı algoritmalar 5 katlı çapraz doğrulama ile karşılaştırılmıştır.

---

### 🗂️ İçerik

* [Veri Kümesi](#veri-kümesi)
* [Kullanılan Yöntemler](#kullanılan-yöntemler)
* [Modeller ve Karşılaştırma](#modeller-ve-karşılaştırma)
* [Görselleştirmeler](#görselleştirmeler)
* [Gereksinimler](#gereksinimler)
* [Çalıştırma](#çalıştırma)
* [Sonuçlar](#sonuçlar)

---

### 📆 Veri Kümesi

Kullanılan veri kümesi: [`bbc-text.csv`](https://www.kaggle.com/datasets/moazeldsokyx/bbc-news)

Veri kümesi aşağıdaki kategorilere ait toplam 2225 haber içermektedir:

* business
* entertainment
* politics
* sport
* tech

---

### ⚖️ Kullanılan Yöntemler

* **TF-IDF Vektörleştirme**: 5000 öznitelik ile metinler sayısal hale getirildi.
* **Label Encoding**: Kategoriler sayısal etiketlere dönüştürüldü.
* **Çapraz Doğrulama (StratifiedKFold)**: Denge korunarak 5 katlı değerlendirme.
* **Performans Metrikleri**:

  * Accuracy
  * Precision (macro)
  * Recall (macro)
  * Specificity
  * F1-score
  * Matthews Correlation Coefficient (MCC)
  * Confusion Matrix

---

### 🤖 Modeller ve Karşılaştırma

Projede aşağıdaki makine öğrenmesi algoritmaları değerlendirilmiştir:

* Naive Bayes
* Logistic Regression
* Linear SVM
* Random Forest
* k-Nearest Neighbors (kNN)
* Multi-Layer Perceptron (MLP)
* Decision Tree
* XGBoost

---

### 📊 Görselleştirmeler

1. **Performans Matrisi**
   Her modelin doğruluk, f1, precision gibi metriklerine göre genel bir kıyaslama.

2. **Karmaşıklık Matrisleri (Confusion Matrix)**
   Her model için gerçek ve tahmin edilen sınıflar arasındaki ilişkileri gösterir.

3. **Sınıf Bazlı Metin Performans Görseli**
   Logistic Regression modeli kullanılarak her bir sınıf için precision / recall / f1-score karşılaştırılmıştır.

---


---

### 🚀 Çalıştırma

```bash
python main.py
```

Not: `bbc-text.csv` dosyası çalışma dizininde olmalıdır.

---

### 📈 Sonuçlar

* En yüksek başarı Logistic Regression, SVM ve XGBoost modellerinde görülmüştür.
* Logistic Regression modeli sınıflar arasında dengeli bir performans sergilemiştir.
* Projenin tamamı yaklaşık **X saniye** içinde çalışmaktadır.
  *(Çalışma süreci: `Toplam Çalışma Süreci` çıktısından alınabilir.)*

---

### 📝 Lisans

Bu proje eğitim ve araştırma amaçlıdır. BBC veri kümesi yalnızca açık kaynak analiz içindir.
