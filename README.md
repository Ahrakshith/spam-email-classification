# Spam / Ham Email Classification

## 📌 Overview
This project is a **Spam/Ham email classification system** built using Jupyter Notebook.  
It trains a machine learning pipeline to classify emails/messages as **spam** or **ham** using  
**Count/TF-IDF Vectorizer** and **Multinomial Naive Bayes (MNB)**.  

The project also includes a simple **Streamlit interface** for interactive predictions.

---

## ⚙️ Technology Used
- **Python (Jupyter Notebook)**
- **pandas, numpy** → data loading & preprocessing
- **scikit-learn** → CountVectorizer / TfidfVectorizer, MultinomialNB, Pipeline, evaluation
- **joblib** → saving/loading trained model
- **nltk** → optional stopword handling
- **matplotlib, seaborn** → plots & confusion matrix
- **Streamlit** → simple demo app

---

## 🔄 Process
1. **Dataset** → Load `spam_ham_database.csv` (label + text columns).  
2. **Preprocessing** → lowercase, remove URLs/emails, normalize numbers, clean special characters (lemmatization optional).  
3. **Train/Test Split** → 80% training, 20% testing with stratification to keep class balance.  
4. **Vectorization** → Convert text into numeric features using CountVectorizer / TF-IDF with unigrams + bigrams.  
5. **Model Training** → Multinomial Naive Bayes classifier trained on the vectorized data.  
6. **Evaluation** → Accuracy, Precision, Recall, F1-score, and Confusion Matrix.  
7. **Artifacts** → Save the trained pipeline with `joblib` for reuse.  
8. **Streamlit Demo** → Input text → cleaned → classified as Spam/Ham with probability.

---

## ✅ Outcome
- Achieved high accuracy (~90–95% depending on preprocessing).  
- Built a **reproducible ML pipeline** (preprocessing + vectorizer + model).  
- Demonstrated results interactively with **Streamlit**.  
- Clear separation of stages: preprocessing → training → evaluation → serving.

---

## 🖥️ Streamlit Interface
The Streamlit app provides a simple way to test the model.

### Run locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Train & save pipeline (run the notebook first if needed)
jupyter notebook notebook.ipynb

# Run Streamlit demo
streamlit run streamlit_app.py
