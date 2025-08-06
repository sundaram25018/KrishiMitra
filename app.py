from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv, find_dotenv
from utils.crop_recommender import get_crop_recommendations
from utils.timetable import generate_timetable
from utils.chatbot import qa_chain
import google.generativeai as genai
import numpy as np
import pickle
from PIL import Image
import os
import requests

# Load environment
load_dotenv(find_dotenv())

# Flask config
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ML models
model = pickle.load(open('models/model.pkl', 'rb'))
sc = pickle.load(open('models/scaler.pkl', 'rb'))
lb = pickle.load(open('models/label.pkl', 'rb'))

NEWS_API_KEY = "43463497e7164948abcac068d62df017"

# Routes
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
        data = res.json()
        articles = data['articles']
        for article in articles:
            article['image'] = article.get('urlToImage', 'default.png')
    except Exception as e:
        articles = [{"title": "Failed to fetch news", "description": str(e), "url": "#", "image": "default.png"}]
    return render_template("news.html", articles=articles)

@app.route("/timetable", methods=["GET", "POST"])
def crop_timetable():
    output = None
    if request.method == "POST":
        crop = request.form.get("crop")
        soil = request.form.get("soil")
        region = request.form.get("region")
        date = request.form.get("start_date")
        output = generate_timetable(crop, soil, region, date)
    return render_template("timetable.html", output=output)

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
        prompt = """
You are an expert agricultural advisor and plant pathologist.
Identify the plant disease in this image and provide practical treatment and prevention advice.
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt, image])
        diagnosis = response.text.strip()
    return render_template("chat.html", diagnosis=diagnosis, image_filename=image_filename)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = [float(request.form[key]) for key in ['Nitrogen', 'Phosporus', 'Potassium', 'Temperature', 'Humidity', 'Ph', 'Rainfall']]
        scaled = sc.transform(np.array(data).reshape(1, -1))
        prediction = model.predict(scaled)
        crop_name = lb.inverse_transform(prediction)[0]
        return render_template('index.html', result=f"{crop_name.capitalize()} is the best crop to be cultivated right there.")
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

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

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
