from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import json
import google.generativeai as genai

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import uuid
import difflib

app = Flask(__name__)
CORS(app)

# Load the dataset - Updated to use relative path for deployment
dataset_path = os.path.join(os.getcwd(), 'Updated_symbipredict.csv')
df = pd.read_csv(dataset_path)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Identify symptom columns (all except last 2: prognosis, medication)
symptom_columns = df.columns[:-2]
# Updated to use applymap instead of map for better compatibility
df[symptom_columns] = df[symptom_columns].applymap(lambda x: str(x).strip().lower() in ['yes', '1'])

# Description folder
# Ensure this folder exists and contains your disease description JSON files
description_folder = os.path.join(os.getcwd(), "disease_descriptions")

# Configure Gemini API (YOUR API KEY GOES HERE)
# It's best practice to use environment variables for API keys in production
# For development, you can hardcode it here, but be very careful
genai.configure(api_key='AIzaSyDXFhlMqSlENMbj31MKPbrM4lloHkRc20M') # <--- REPLACE WITH YOUR ACTUAL GEMINI API KEY

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    # Optionally, you can also pass context (selected symptoms, predictions) from frontend
    # For a more complex chat, you might send chat history here too.
    
    if not user_message:
        return jsonify({'response': 'Please provide a message.'}), 400

    # Modified line
    model = genai.GenerativeModel('gemini-2.0-flash')

    try:
        # For a basic chat, just send the user message
        # For a more context-aware bot, you'd build a more complex prompt here
        # incorporating predictions, selected symptoms, and chat history.
        
        # Example of adding some basic context to the prompt for Gemini:
        context_info = ""
        if 'selected_symptoms' in data and data['selected_symptoms']:
            context_info += f"User's reported symptoms: {', '.join(data['selected_symptoms'])}.\n"
        if 'last_predictions' in data and data['last_predictions']:
            context_info += "Previous disease predictions for user:\n"
            for pred in data['last_predictions']:
                context_info += f"- {pred['prognosis']} (Confidence: {pred['confidence_score']*100:.0f}%)\n"
        
        full_prompt = f"""You are a helpful and informative medical AI assistant. Your goal is to provide general health information and explanations based on symptoms or predictions. Always remind the user to consult a healthcare professional for accurate diagnosis and treatment.

{context_info}
User's question: {user_message}

Please provide a helpful, informative response. Remember to include the medical disclaimer."""

        response = model.generate_content(full_prompt)
        return jsonify({'response': response.text})
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({'response': 'Sorry, I am currently unable to process your request. Please try again later.'}), 500

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
            # Use a slightly lower cutoff for more flexibility in matching common typos
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
        # Remove duplicates by prognosis - Updated to use simpler approach from second code
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
        # Log the error for debugging
        print(f"Prediction error: {e}")
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
            # Check if there's enough space for the next entry
            required_space = 20 + 15 + 30 + 40  # Heading + medication + description + tips
            if y < (50 + required_space): # Leave some margin at bottom
                c.showPage()
                y = height - 50
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, y, "Disease Prediction Report (Cont.)")
                y -= 30
                c.setFont("Helvetica", 10)
                c.drawString(50, y, f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
                y -= 30

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, f"{idx}. {result['prognosis']}")
            y -= 20

            c.setFont("Helvetica", 10)
            # Use textobject for multi-line text to handle wrapping if descriptions are long
            # Simple direct drawing for now as descriptions are assumed to fit
            c.drawString(60, y, f"Medication: {result.get('medication', 'N/A')}")
            y -= 15
            
            # For longer text, consider using textobject for multi-line support
            description_lines = []
            desc = result.get('description', 'N/A')
            # Basic wrapping for description
            max_desc_width = width - 120 # 50 (left margin) + 60 (indent) = 110. Let's use 120 for safety
            temp_desc = ""
            for word in desc.split(' '):
                if c.stringWidth(temp_desc + word, "Helvetica", 10) < max_desc_width:
                    temp_desc += word + " "
                else:
                    description_lines.append(temp_desc.strip())
                    temp_desc = word + " "
            if temp_desc:
                description_lines.append(temp_desc.strip())

            c.drawString(60, y, "Description:")
            y -= 15
            for line in description_lines:
                c.drawString(70, y, line)
                y -= 12 # Line spacing for description

            y -= 10 # Extra space between description and tips

            tips_lines = []
            tips = result.get('tips', 'N/A')
            # Basic wrapping for tips
            temp_tips = ""
            for word in tips.split(' '):
                if c.stringWidth(temp_tips + word, "Helvetica", 10) < max_desc_width: # Use same max width
                    temp_tips += word + " "
                else:
                    tips_lines.append(temp_tips.strip())
                    temp_tips = word + " "
            if temp_tips:
                tips_lines.append(temp_tips.strip())

            c.drawString(60, y, "Tips:")
            y -= 15
            for line in tips_lines:
                c.drawString(70, y, line)
                y -= 12 # Line spacing for tips

            y -= 40 # Space after each prediction block

        c.save()

        # Return the filename so frontend can construct download URL
        return jsonify({'filename': filename})

    except Exception as e:
        print(f"PDF generation error: {e}") # Log error on server side
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the description folder exists
    if not os.path.exists(description_folder):
        os.makedirs(description_folder)
        print(f"Created disease description folder: {description_folder}")

    # Updated to support deployment with PORT from environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True) # debug=True is good for development, disable in production