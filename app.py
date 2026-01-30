import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.messages import HumanMessage

# =========================================================
# 🔐 LOAD API KEY FROM STREAMLIT SECRETS (OPTION 1)
# =========================================================
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# =========================================================
# ⚙️ PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MediMind Pro",
    layout="wide"
)

st.title("🧠 MediMind Pro – AI Mental Health Platform")

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
# 🤖 TRAIN / LOAD MODEL
# =========================================================
@st.cache_resource
def train_model():
    data = df.copy()
    le = LabelEncoder()

    for col in data.select_dtypes(include="object").columns:
        data[col] = le.fit_transform(data[col].astype(str))

    target = data.columns[-1]
    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "mental_health_model.pkl")
    return model, X_test, y_test, X

model, X_test, y_test, X_all = train_model()

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
# 💬 PDF-BASED AI CHATBOT (RAG)
# =========================================================
elif menu == "AI Chatbot (PDF)":
    st.subheader("💬 Mental Health Assistant (PDF-Grounded)")

    @st.cache_resource
    def load_pdf_vectorstore():
        reader = PdfReader("mental_health_Document.pdf")
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        return FAISS.from_texts(chunks, embeddings)

    vectorstore = load_pdf_vectorstore()

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.2
    )

    query = st.text_input("Ask a mental health question:")

    if query:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        You are a mental health assistant.
        Answer ONLY using the context below.

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
    st.subheader("📘 About MediMind Pro")
    st.markdown("""
    **MediMind Pro** is an end-to-end AI mental health platform featuring:

    - Machine Learning risk prediction
    - SHAP explainability
    - Secure authentication
    - PDF-grounded AI chatbot (RAG)
    - Production-ready Streamlit deployment

    **Tech Stack:**  
    Python · Streamlit · Scikit-Learn · SHAP · LangChain · FAISS · Gemini
    """)

