# 🌾 KrishiMitra – Smart Farming Assistant Web App

KrishiMitra is an intelligent farming assistant designed to empower Indian farmers with real-time support in the field. It provides crop recommendations, AI-powered chatbot help, weather forecasts, latest agriculture news, and access to government schemes — all in one simple platform.

---

## 🚀 Features

✅ **💬 Chatbot Assistant**  
Ask farming-related queries and get instant advice (powered by rule-based logic or AI).

✅ **🌦️ Weather Dashboard**  
Enter your State & District to fetch live weather updates via OpenWeather API.

✅ **📰 Agri News Feed**  
Shows latest agricultural news from trusted sources like NDTV and DownToEarth.

✅ **📢 Government Schemes**  
Displays up-to-date government schemes using verified official links (PM-KISAN, AIF, eNAM etc.)

✅ **🌱 Crop Recommendation **  
 ML-based crop prediction based on soil, climate and rainfall data and .

 ✅ **🌱 Disease Prediction **  
 ML-based Disease prediction by uploading an image and suggest a discription and solutions .

---

## 💻 Tech Stack

- **model training**: Machine learning, scikit-learn
- **Frontend**: HTML, Bootstrap 5, JavaScript
- **Backend**: Python + Flask
- **APIs**:
  - OpenWeatherMap (weather)
  - Feedparser (news)
  - Static or scraped scheme data
- **Optional AI**: Gemini/OpenAI GPT or Rule-based JSON logic

---

## 📂 Project Structure

.
├── models/                  
│   ├── model.pkl             
│   ├── scaler.pkl            
│   ├── label.pkl            

├── Notebook/                  
│   ├── crop.ipynb            
│   ├── crop-recomendation.csv            
│   

├── static/                  
│   ├── style/
│        ├──index.css
│        ├──chat.css
│        ├──news.css
│        ├──diagnous.css
│        ├──weather.css
│
│   ├── Upload/           
│       

├── templates/                  
│   ├── index.html              
│   ├── chat.html             
│   ├── weather.html            
│   ├── news.html                            

├── app.py       
├── README.md                    
└── requirements.txt  



---

## Demo 

![image](https://github.com/user-attachments/assets/572bb488-b753-4d2d-ac49-0705d35d2eb9)
![image](https://github.com/user-attachments/assets/f4619f71-8ae2-49a6-a3aa-137096e7f106)
![image](https://github.com/user-attachments/assets/1c7cc827-0220-4ac7-912d-0eb62833ec44)
![image](https://github.com/user-attachments/assets/669762d1-4b66-4184-b8c3-6c0644c6921f)





## 🔧 Setup Instructions

1. Clone the repo  
   `git clone https://github.com/sundaram25018/krishimitra.git`

2. Install requirements  
   `pip install requirements.txtr`

3. Add your API keys:
   - Weather API: OpenWeatherMap
   - (Optional) Gemini/OpenAI API

4. Run the Flask app  
   `python app.py`

5. Visit  
   `http://localhost:5000`

---

## 📌 To-Do (Next Features)

- [ ] Language support: Hindi, Marathi
- [ ] Voice-to-text integration
- [ ] SMS alerts for schemes/weather
- [ ] AI Model fallback offline

---

## 🤝 Contributing

Want to help build India's smartest agri assistant? Fork the repo, suggest enhancements, and contribute!

---

## 🛡 License

MIT License

---

## 🙏 Acknowledgements

- Government of India portals
- OpenWeatherMap API
- NDTV RSS Feed
- Farmers across India 🇮🇳


