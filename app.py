import base64
import io
import json
import os
import re

from flask import Flask, jsonify, render_template, request
from groq import Groq
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Initialize Groq Client
def get_groq_client():
    """Dynamically get or initialize the Groq client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return Groq(api_key=api_key)
    return None


@app.route('/')
def index():
    """Render the landing page."""
    return render_template('index.html')


@app.route('/features')
def features():
    """Render the features page."""
    return render_template('features.html')


@app.route('/Howitworks')
def how_it_works():
    """Render the how it works page."""
    return render_template('Howitworks.html')


@app.route('/analyze')
def analyze():
    """Render the analyze page."""
    return render_template('analyze.html')


@app.route('/results')
def results():
    """Render the results page."""
    return render_template('results.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Handle X-ray image analysis request using Groq's Vision API."""
    if 'xray' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['xray']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        client = get_groq_client()
        if not client:
            return jsonify({'error': 'Groq API key not set. Please set the GROQ_API_KEY environment variable.'}), 500

        # Read image bytes and convert to RGB/JPEG for processing
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))

        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=85)
        buf.seek(0)
        image_bytes = buf.getvalue()

        # Base64 encode for Groq API
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        prompt = """You are an expert radiologist AI assistant. Analyze this X-ray image thoroughly.
        Provide your response ONLY as a valid JSON object with no extra text.

        Use exactly this structure:
        {
          "overall_status": "Normal or Abnormal or Requires Attention",
          "confidence": 85,
          "findings": [
            {
              "name": "Finding name",
              "region": "Anatomical region",
              "severity": "Normal or Mild or Moderate or Severe",
              "confidence": 80,
              "description": "Detailed description of the finding"
            }
          ],
          "differential_diagnosis": ["Condition 1", "Condition 2"],
          "recommendations": ["Recommendation 1"],
          "summary": "Overall summary paragraph",
          "image_quality": "Good or Fair or Poor"
        }"""

        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            model=GROQ_MODEL,
            response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content

        # Robust JSON extraction
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = build_fallback_result(response_text)
        else:
            result = build_fallback_result(response_text)

        return jsonify({'success': True, 'result': result})

    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "401" in error_msg:
            return jsonify({'error': 'Invalid or missing Groq API key. Get one at https://console.groq.com/'}), 500
        return jsonify({'error': error_msg}), 500


def build_fallback_result(text):
    """Fallback mechanism for non-JSON responses."""
    return {
        "overall_status": "Analysis Complete",
        "confidence": 75,
        "findings": [
            {
                "name": "AI Analysis",
                "region": "Full Image",
                "severity": "Reviewed",
                "confidence": 75,
                "description": text[:500] if text else "Analysis completed."
            }
        ],
        "differential_diagnosis": [],
        "recommendations": ["Consult with a qualified radiologist for clinical interpretation"],
        "summary": text if text else "Analysis completed successfully.",
        "image_quality": "Good"
    }


@app.route('/api/status', methods=['GET'])
def check_status():
    """Check the status of the Groq API connection."""
    client = get_groq_client()
    if not client:
        return jsonify({'running': False, 'message': 'API key not set. Add your Groq API key.'})
    return jsonify({'running': True, 'model': GROQ_MODEL, 'message': 'Groq API ready'})


if __name__ == '__main__':
    print("=" * 50)
    print("  RadiantAI — X-Ray Analysis (Groq Mode)")
    print("=" * 50)
    print("  Set it via: set GROQ_API_KEY=your_key_here")
    print("  Open browser at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)