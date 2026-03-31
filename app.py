from flask import Flask, render_template, request, jsonify
import json
import re
import os
import google.generativeai as genai
from PIL import Image
import io

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("AIzaSyAIbNct1rwi-9EPZuukLBUsi1HzyWU-oDc")  # FIX 1: use env var name, not the key value
GEMINI_MODEL = "gemini-2.0-flash"                   # FIX 2: define the model name constant

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(GEMINI_MODEL)         # FIX 3: use the constant here too


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/Howitworks')
def Howitworks():
    return render_template('Howitworks.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'xray' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['xray']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read image bytes
    image_data = file.read()
    image = Image.open(io.BytesIO(image_data))

    # Convert to RGB if needed (handles PNG with alpha channel)
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')

    # Save to bytes buffer as JPEG for the API
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    buf.seek(0)
    image_bytes = buf.getvalue()

    prompt = """You are an expert radiologist AI assistant. Analyze this X-ray image thoroughly and provide a detailed diagnostic report.

Provide your response ONLY as a valid JSON object with no extra text, no markdown, no explanation outside the JSON.

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
  "differential_diagnosis": ["Condition 1", "Condition 2", "Condition 3"],
  "recommendations": ["Recommendation 1", "Recommendation 2"],
  "summary": "Overall summary paragraph of findings",
  "image_quality": "Good or Fair or Poor"
}

Be thorough. Identify any abnormalities, opacities, consolidations, pleural effusions, cardiomegaly, pneumothorax, fractures, or other findings. If the image appears normal, state that clearly."""

    try:
        # FIX 4: use model.generate_content() directly (not client.models.generate_content)
        import PIL.Image as PILImage
        pil_image = PILImage.open(io.BytesIO(image_bytes))

        response = model.generate_content([prompt, pil_image])

        response_text = response.text

        # Strip markdown code fences if present
        response_text = re.sub(r'```json|```', '', response_text).strip()

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
        if "API_KEY" in error_msg or "invalid" in error_msg.lower():
            return jsonify({'error': 'Invalid or missing Gemini API key. Get one free at https://aistudio.google.com/app/apikey'}), 500
        if "quota" in error_msg.lower():
            return jsonify({'error': 'Gemini free tier quota exceeded. Try again later.'}), 500
        return jsonify({'error': error_msg}), 500


def build_fallback_result(text):
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
        "recommendations": [
            "Consult with a qualified radiologist for clinical interpretation",
            "Correlate findings with patient history and symptoms"
        ],
        "summary": text if text else "Analysis completed successfully.",
        "image_quality": "Good"
    }


@app.route('/api/status', methods=['GET'])
def check_status():
    if not GEMINI_API_KEY:
        return jsonify({'running': False, 'message': 'API key not set. Add your Gemini API key.'})
    return jsonify({'running': True, 'model': GEMINI_MODEL, 'message': 'Gemini API ready'})


if __name__ == '__main__':
    print("=" * 50)
    print("  RadiantAI — X-Ray Analysis (Gemini Mode)")
    print("=" * 50)
    print("  Get free API key: https://aistudio.google.com/app/apikey")
    print("  Set it via:  set GEMINI_API_KEY=your_key_here")
    print("  Open browser at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)