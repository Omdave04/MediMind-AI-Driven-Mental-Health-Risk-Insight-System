import os
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from pypdf import PdfReader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.messages import HumanMessage

# =========================================================
# 🔐 LOAD API KEY FROM STREAMLIT SECRETS
# =========================================================
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# =========================================================
# ⚙️ PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MediMind – Mental Health Intelligence",
    layout="wide"
)

st.title("🧠 MediMind – Explainable AI Platform for Mental Health Intelligence")

# =========================================================
# 🔐 AUTHENTICATION
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

    st.stop()

# =========================================================
# 📌 SIDEBAR
# =========================================================
menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "ML Prediction",
        "Explainability (SHAP)",
        "AI Chatbot (PDF)",
        "About"
    ]
)

# =========================================================
# 📊 LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("Mental Health Dataset.csv")

df = load_data()

# =========================================================
# 🤖 TRAIN MODEL (NO PKL, TRAIN-ONCE)
# =========================================================
@st.cache_resource
def train_model():
    data = df.copy()
    label_encoders = {}

    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    target = data.columns[-1]
    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model, X_test, X

model, X_test, X_all = train_model()

# =========================================================
# 📊 DASHBOARD
# =========================================================
if menu == "Dashboard":
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

# =========================================================
# 🤖 ML PREDICTION
# =========================================================
elif menu == "ML Prediction":
    st.subheader("🤖 Mental Health Risk Prediction")

    user_input = {}
    for col in X_all.columns:
        user_input[col] = st.number_input(
            col,
            float(X_all[col].min()),
            float(X_all[col].max())
        )

    if st.button("Predict Risk"):
        user_df = pd.DataFrame([user_input])
        prediction = model.predict(user_df)[0]

        if prediction == 1:
            st.error("⚠ High Risk Detected – Professional support recommended")
        else:
            st.success("✅ Low Risk Detected")

# =========================================================
# 🔍 SHAP EXPLAINABILITY
# =========================================================
elif menu == "Explainability (SHAP)":
    st.subheader("🔍 Model Explainability (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

# =========================================================
# 💬 AI CHATBOT (PDF – SAFE VERSION)
# =========================================================
elif menu == "AI Chatbot (PDF)":
    st.subheader("💬 Mental Health Assistant (PDF-Grounded)")

    # ---------- Simple text splitter (PURE PYTHON) ----------
    def simple_text_splitter(text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        return chunks

    # ---------- Load PDF & Create Vector Store ----------
    @st.cache_resource
    def load_pdf_vectorstore():
        reader = PdfReader("mental_health_Document.pdf")
        full_text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        chunks = simple_text_splitter(full_text)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        return FAISS.from_texts(chunks, embeddings)

    vectorstore = load_pdf_vectorstore()

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.2
    )

    query = st.text_input("Ask a mental health question based on the document:")

    if query:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are a mental health assistant.
        Answer ONLY using the context below.
        If the answer is not found, say so clearly.

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        st.info(response.content)

# =========================================================
# 📘 ABOUT
# =========================================================
else:
    st.subheader("📘 About MediMind")
    st.markdown("""
    **MediMind** is an explainable AI-powered mental health intelligence platform.

    **Key Features**
    - ML-based mental health risk prediction
    - SHAP explainability for transparency
    - Secure authentication
    - PDF-grounded AI assistant (RAG)
    - Streamlit Cloud deployment-ready

    **Tech Stack**
    Python · Streamlit · Scikit-learn · SHAP · FAISS · Gemini
    """)

