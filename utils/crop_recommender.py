import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_crop_recommendations(country, location, season):
    prompt = f"""
You are an expert agricultural advisor. Recommend 5 crops for:
Country: {country}
Location: {location}
Season: {season}

Include:
1. Crop name
2. Reason for recommendation
3. Yield expectation
4. Market potential
5. Farming tips

Be practical and clear.
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model.generate_content(prompt).text.strip()
