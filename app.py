from flask import Flask, request, jsonify, render_template_string
import cv2
import imutils
import numpy as np
import math
from imutils import contours

app = Flask(__name__)

# --- THE FRONTEND (HTML + Javascript) ---
# This is the website the user sees when they visit the app
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LigneLens Web App</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen font-sans">
    <div class="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md text-center">
        <h1 class="text-3xl font-extrabold text-gray-900 mb-2">Ligne<span class="text-blue-600">Lens</span></h1>
        <p class="text-gray-500 mb-8">Upload a photo of your button and coin.</p>

        <form id="uploadForm" class="space-y-6">
            <div class="border-2 border-dashed border-gray-300 rounded-xl p-6 hover:border-blue-500 transition">
                <input type="file" id="imageInput" accept="image/*" required class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100">
            </div>

            <div>
                <label class="block text-left text-sm font-semibold text-gray-700 mb-2">Reference Coin</label>
                <select id="coinSize" class="w-full border border-gray-300 rounded-lg px-4 py-3 bg-gray-50">
                    <option value="23.0">₹5 Coin (23.0 mm)</option>
                    <option value="21.93">₹1 Coin (21.93 mm)</option>
                    <option value="24.26">US Quarter (24.26 mm)</option>
                </select>
            </div>

            <button type="submit" class="w-full bg-blue-600 text-white font-bold py-3 rounded-xl hover:bg-blue-700 transition">
                Calculate Size
            </button>
        </form>

        <div id="loading" class="hidden mt-6 text-blue-600 font-bold">Processing AI Vision...</div>

        <div id="resultBox" class="hidden mt-8 p-6 bg-green-50 border border-green-200 rounded-xl">
            <h2 class="text-green-800 text-sm font-bold uppercase tracking-wide">Measurement</h2>
            <div id="ligneResult" class="text-5xl font-black text-green-600 mt-2">--L</div>
            <div id="mmResult" class="text-gray-600 mt-1 font-medium">-- mm</div>
        </div>
        
        <div id="errorBox" class="hidden mt-6 p-4 bg-red-50 text-red-600 border border-red-200 rounded-xl font-medium"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const imageFile = document.getElementById('imageInput').files[0];
            const coinSize = document.getElementById('coinSize').value;
            
            if (!imageFile) return;

            // Show loading, hide results
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('resultBox').classList.add('hidden');
            document.getElementById('errorBox').classList.add('hidden');

            // Package the file and data to send to Python
            const formData = new FormData();
            formData.append('image', imageFile);
            formData.append('reference_mm', coinSize);

            try {
                // Send it to our Python backend
                const response = await fetch('/measure', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                document.getElementById('loading').classList.add('hidden');

                if (data.error) {
                    document.getElementById('errorBox').textContent = data.error;
                    document.getElementById('errorBox').classList.remove('hidden');
                } else {
                    document.getElementById('ligneResult').textContent = data.ligne_size + "L";
                    document.getElementById('mmResult').textContent = data.diameter_mm + " mm";
                    document.getElementById('resultBox').classList.remove('hidden');
                }
            } catch (err) {
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('errorBox').textContent = "Server error. Is the backend running?";
                document.getElementById('errorBox').classList.remove('hidden');
            }
        });
    </script>
</body>
</html>
"""

# --- THE BACKEND (Python CV Logic) ---

def calculate_button_size(image_bytes, reference_width_mm):
    # Convert the uploaded web image bytes into a format OpenCV can read
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Could not read the uploaded image."}

    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    valid_circles = []
    
    for c in cnts:
        actual_area = cv2.contourArea(c)
        if actual_area < 150: 
            continue
            
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius == 0: continue
            
        perfect_circle_area = math.pi * (radius ** 2)
        fill_ratio = actual_area / perfect_circle_area
        
        # The Fill Ratio Test we perfected
        if fill_ratio > 0.75:
            valid_circles.append(c)

    if len(valid_circles) < 2:
        return {"error": f"Found {len(valid_circles)} round objects. We need exactly 1 coin and 1 button in the frame."}

    valid_circles = sorted(valid_circles, key=cv2.contourArea, reverse=True)[:2]
    valid_circles, _ = contours.sort_contours(valid_circles)

    pixels_per_metric = None
    final_ligne = 0
    final_mm = 0

    for c in valid_circles:
        # Calculate real width using total Area to ignore edge shadows
        actual_area = cv2.contourArea(c)
        math_radius = math.sqrt(actual_area / math.pi)
        pixel_width = math_radius * 2

        # Left object (The Coin)
        if pixels_per_metric is None:
            pixels_per_metric = pixel_width / reference_width_mm
            continue

        # Right object (The Button)
        button_width_mm = pixel_width / pixels_per_metric
        button_ligne = button_width_mm / 0.635
        
        final_mm = round(button_width_mm, 1)
        final_ligne = round(button_ligne)

    return {
        "ligne_size": final_ligne,
        "diameter_mm": final_mm
    }


# --- THE ROUTES (Connecting Web to Python) ---

@app.route('/')
def home():
    # Serve the HTML page when someone visits the main URL
    return render_template_string(HTML_PAGE)

@app.route('/measure', methods=['POST'])
def measure():
    # Handle the image uploaded by the user
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    reference_mm = float(request.form.get('reference_mm', 23.0))
    
    # Read the image directly from memory (no saving to hard drive required)
    image_bytes = file.read()
    
    # Run our computer vision engine
    result = calculate_button_size(image_bytes, reference_mm)
    
    # Send the JSON result back to the website
    return jsonify(result)

if __name__ == '__main__':
    # Start the local web server
    print("\n--- LIGNELENS WEB APP IS RUNNING ---")
    print("Open your browser and go to: http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=8080)
