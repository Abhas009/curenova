from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import uuid
import difflib
import os

app = Flask(__name__)
CORS(app)

# Load the dataset
dataset_path = os.path.join(os.getcwd(), 'Updated_symbipredict.csv')
df = pd.read_csv(dataset_path)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Identify symptom columns (all except last 2: prognosis, medication)
symptom_columns = df.columns[:-2]
df[symptom_columns] = df[symptom_columns].applymap(lambda x: str(x).strip().lower() in ['yes', '1'])

# Description folder
description_folder = os.path.join(os.getcwd(), "disease_descriptions")

@app.route('/')
def home():
    return "Disease Prediction API is running. Use the '/predict' endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'Symptoms not provided'}), 400

        # Get valid symptoms from dataset
        valid_symptoms = list(symptom_columns)

        # Clean and autocorrect user symptoms
        user_symptoms = set()
        raw_input = data['symptoms']

        for symptom in raw_input:
            symptom = symptom.strip().lower()
            matches = difflib.get_close_matches(symptom, valid_symptoms, n=1, cutoff=0.7)
            if matches:
                user_symptoms.add(matches[0])
            else:
                user_symptoms.add(symptom)  # fallback to original if no match found

        match_results = []

        for _, row in df.iterrows():
            # Get symptoms that are True for this disease
            disease_symptoms = set(row[symptom_columns][row[symptom_columns]].index)
            match_score = len(user_symptoms.intersection(disease_symptoms))
            total_disease_symptoms = len(disease_symptoms)
            confidence_score = round(match_score / total_disease_symptoms, 2) if total_disease_symptoms else 0

            if match_score > 0:
                prognosis = row['prognosis']
                medication = row['medication']
                file_name = prognosis.strip().lower().replace(" ", "_") + ".json"
                file_path = os.path.join(description_folder, file_name)

                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    # Language support
                    language = data.get('language', '').lower()
                    if language and 'translations' in info and language in info['translations']:
                        description = info['translations'][language].get("description", info.get("description", "No description available."))
                        tips = info['translations'][language].get("tips", info.get("tips", "No tips available."))
                    else:
                        description = info.get("description", "No description available.")
                        tips = info.get("tips", "No tips available.")
                else:
                    description = "No description available."
                    tips = "No tips available."

                match_results.append({
                    "prognosis": prognosis,
                    "medication": medication,
                    "description": description,
                    "tips": tips,
                    "match_score": match_score,
                    "confidence_score": confidence_score
                })

        # Sort by confidence_score (instead of match_score)
        # Remove duplicates by prognosis
        seen = set()
        unique_results = []
        for result in sorted(match_results, key=lambda x: x['confidence_score'], reverse=True):
            if result['prognosis'] not in seen:
                seen.add(result['prognosis'])
                unique_results.append(result)

        # Filter top 3 based on confidence threshold
        top_matches = [match for match in unique_results if match['confidence_score'] >= 0.3][:3]

        if top_matches:
            return jsonify(top_matches)
        else:
            return jsonify({'message': 'No strong prediction. Please enter more symptoms.'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptom_list():
    return jsonify(sorted(symptom_columns.tolist()))

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        predictions = data.get('predictions', [])

        if not predictions:
            return jsonify({'error': 'No predictions provided'}), 400

        # Create unique filename
        filename = f"prediction_report_{uuid.uuid4().hex[:8]}.pdf"
        filepath = os.path.join(os.getcwd(), filename)

        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        y = height - 50

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Disease Prediction Report")
        y -= 30

        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
        y -= 30

        for idx, result in enumerate(predictions[:3], start=1):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, f"{idx}. {result['prognosis']}")
            y -= 20

            c.setFont("Helvetica", 10)
            c.drawString(60, y, f"Medication: {result.get('medication', 'N/A')}")
            y -= 15
            c.drawString(60, y, f"Description: {result.get('description', 'N/A')}")
            y -= 30
            c.drawString(60, y, f"Tips: {result.get('tips', 'N/A')}")
            y -= 40

            if y < 100:
                c.showPage()
                y = height - 50

        c.save()

        return jsonify({'filename': filename})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)