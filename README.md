#📘 README.md

## 🛒 Product Category Classifier

This project provides an automated product category classification system based solely on product titles. The model enables fast and accurate categorization of items for online stores, significantly reducing manual work and improving data entry efficiency.

The project includes:

- detailed exploratory data analysis (EDA)

- data cleaning

- feature engineering

- model training and evaluation

- saving and loading ML pipeline components

- production-ready training and prediction scripts

---

## 📂 Project Structure

```graphql
product-category-classifier/
│
├── data/
│   ├── products.csv                     # Raw dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA and data cleaning
│   └── 02_feature_engineering_and_modeling.ipynb 
│                                        # Feature engineering, training, evaluation
│
├── models/
│   ├── product_classifier.pkl           # Final Linear SVC model
│   ├── scaler.pkl                       # MinMaxScaler object
│   ├── label_encoder.pkl                # LabelEncoder for categories
│   └── tfidf_vectorizer.pkl             # TF-IDF Vectorizer
│
├── src/
│   ├── train_model.py                   # Training & serialization script
│   └── predict_category.py              # CLI prediction script
│
└── README.md                            # Project documentation
```
---

# 🚀 Installation & Setup

###1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/product-category-classifier.git
cd product-category-classifier
```

###2️⃣ Install required dependencies

Use your environment or preferred dependency manager (pip, conda, poetry).
Example:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```
---

# 📊 Dataset Description

The dataset used (products.csv) contains the following columns:

- Product ID

- Product Title

- Merchant ID

- Category Label (target variable)

- Product Code

- Number of Views

- Merchant Rating

- Listing Date

Located in:

```bash
data/products.csv
```
--- 

# 🧪 Jupyter Notebooks
### 📘 01_data_exploration.ipynb

Covers:

- dataset inspection

- missing values

- text cleaning

- category distribution

- word-level analysis

- visualization of title length, frequency, etc.

### 📙 02_feature_engineering_and_modeling.ipynb

Includes:

- TF-IDF vectorization

- MinMax scaling

- numeric feature engineering

- label encoding

- model comparison (SVM, Logistic Regression, Naive Bayes, etc.)

- confusion matrix

- final model evaluation

- saving trained components

---- 
# 🤖 Final Production Model

Final selected model: Linear SVC

Saved components:

| Component         | File                     |
| ----------------- | ------------------------ |
| Classifier        | `product_classifier.pkl` |
| TF-IDF Vectorizer | `tfidf_vectorizer.pkl`   |
| Label Encoder     | `label_encoder.pkl`      |
| Scaler            | `scaler.pkl`             |

These components allow fully reproducible predictions.

---
# 🏗 Production Scripts
###🧠 Train the model

```bash
python src/train_model.py
```
This script:

- loads the dataset

- cleans the text

- generates features

- trains the classifier

- evaluates results
- 
- saves model artifacts into models/

---

# 🔍 Predict a product category (CLI)

```bash
python src/predict_category.py
```
Interaction example:

```yamal
Enter product title: iphone 7 32gb gold
➡️ Predicted category: Mobile Phones
```

---

# 📈 Model Evaluation

Evaluation metrics include:

- Accuracy

- Precision / Recall / F1-score

- Confusion Matrix

- Error analysis

Typical performance:
**85–93% accuracy**, depending on feature engineering and preprocessing.

---

# 🧪 Quick Test Examples

| Input Example                 | Expected Category |
| ----------------------------- | ----------------- |
| iphone 7 32gb gold            | Mobile Phones     |
| olympus e m10 mark iii silver | Digital Cameras   |
| kenwood k20mss15 solo         | Microwaves        |
| bosch wap28390gb 8kg          | Washing Machines  |
| bosch serie 4 kgv39vl31g      | Fridge Freezers   |
| smeg sbs8004po                | Fridge Freezers   |

---

# 🛠 Technologies Used

- Python 3.10+

- Scikit-learn

- Pandas

- NumPy

- Matplotlib

- Seaborn

- Joblib

- Jupyter Notebook (GoogleColab)

---

# 📄 License

MIT License — free to use, modify, and distribute.

---

# 🎯 Summary

This project implements a full, production-ready machine learning pipeline for product title classification. It includes data analysis, feature engineering, model training, robust evaluation, component serialization, and a user-friendly CLI prediction interface. It is designed to be clear, maintainable, and easily integrated into real-world e-commerce workflows.