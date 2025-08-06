import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_timetable(crop, soil, region, date):
    prompt = f"""
You are an expert crop planner. Generate a weekly crop timetable for:
Crop: {crop}
Soil: {soil}
Region: {region}
Start Date: {date}

Include weekly:
1. Fertilizer type and quantity
2. Irrigation advice
3. Crop care tasks
4. Weather tips
5. Harvest guidelines

Output clearly by weeks.
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model.generate_content(prompt).text.strip()
