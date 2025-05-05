from flask import Flask, request, render_template,jsonify
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



genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

NEWS_API_KEY = "43463497e7164948abcac068d62df017"  # Replace with your actual News API key


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

# Main
if __name__ == "__main__":
    app.run(debug=True)
