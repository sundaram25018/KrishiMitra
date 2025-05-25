from flask import Flask, request, render_template,jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import json
import google.generativeai as genai
import requests
import numpy as np
import pickle
import feedparser
from PIL import Image
import os

# Load trained model and tools
model = pickle.load(open('models/model.pkl', 'rb'))
sc = pickle.load(open('models/scaler.pkl', 'rb'))
lb = pickle.load(open('models/label.pkl', 'rb'))

# chat bot

DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

NEWS_API_KEY = "43463497e7164948abcac068d62df017"
NEWSDATA_API_KEY = "pub_86819288e6a53d7108b68ee69597c623811ab"  # Replace with your actual News API 
  # Replace with your actual News API key


# Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_crop_recommendations(country, location, season):
    prompt = f"""
You are an expert agricultural advisor. Based on the following details, recommend the 5 most suitable crops to grow:
Country: {country}
Location: {location}
Season or Month: {season}

For each recommended crop, include:
1. Crop name
2. Why it is suitable for the location and season
3. Expected yield range
4. Market demand or selling potential
5. Special tips or considerations for farmers

Respond in a helpful and practical tone.
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/weather')
def weather():
    return render_template("weather.html", api_key="731b31a39cd041de478f62bb934aa935")

@app.route('/news')
def news():
    url = f"https://newsapi.org/v2/everything?q=agriculture+india&sortBy=publishedAt&apiKey={NEWS_API_KEY}&pageSize=12"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        articles = data['articles']
        
        # Ensure articles contain the image URL or a default placeholder image
        for article in articles:
            article['image'] = article.get('urlToImage', 'default.png')  # Replace with a default image URL if no image is found
    except Exception as e:
        articles = [{"title": "Failed to fetch news", "description": str(e), "url": "#", "image": "default_image_url"}]
    
    return render_template("news.html", articles=articles)



@app.route("/diagnose", methods=["GET", "POST"])
def chat():
    recommendations = None
    if request.method == "POST":
        country = request.form["country"]
        location = request.form["location"]
        season = request.form["season"]
        recommendations = get_crop_recommendations(country, location, season)
    return render_template("diagnose.html", recommendations=recommendations)

@app.route("/chat", methods=["GET", "POST"])
def diagnose():
    diagnosis = None
    image_filename = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("chat.html", diagnosis="No file uploaded.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("chat.html", diagnosis="No file selected.")

        image_filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
        file.save(filepath)

        image = Image.open(filepath)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = """
        You are an expert agricultural advisor and plant pathologist.
        I am uploading an image of a diseased plant. Please:
        1. Identify the plant and the likely disease
        2. Explain how the disease affects the plant
        3. Recommend treatment (organic or chemical)
        4. Give prevention tips
        5. State if lab confirmation is needed.
        Be clear and practical.
        """

        response = model.generate_content([prompt, image])
        diagnosis = response.text.strip()

    return render_template("chat.html", diagnosis=diagnosis, image_filename=image_filename)


@app.route("/predict", methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale and predict
        scaled_features = sc.transform(single_pred)
        prediction = model.predict(scaled_features)
        crop_name = lb.inverse_transform(prediction)[0]

        result = f"{crop_name.capitalize()} is the best crop to be cultivated right there."
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")
    

def set_custom_prompt():
    custom_template = """
    Use the pieces of information provided in the context to answer user's question.
    If you don’t know the answer, just say that you don’t know — don’t make it up.
    Don’t provide anything outside of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk.
    """
    return PromptTemplate(template=custom_template, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )

llm = load_llm()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt()}
)

@app.route("/bots")
def bots():
    return render_template("bot.html")

@app.route("/query", methods=["POST"])
def chatbots():
    try:
        query = request.json.get("query")
        if not query:
            return jsonify({"response": "No query provided"}), 400

        result = qa_chain.invoke({"query": query})["result"]
        return jsonify({"response": result + " This is from books and articles, not from the web."})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500


# Main
if __name__ == "__main__":
    app.run(debug=True)
