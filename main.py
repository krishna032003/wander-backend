import os
import glob
import json
import base64
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pypdf import PdfReader
import requests

# Basic text splitting (no langchain dependency)
def split_text(text, chunk_size=1000, chunk_overlap=200):
    """Simple text splitter"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start = end - chunk_overlap if end < text_length else end
        if start <= 0:
            start = end
    
    return chunks

# Initialize environment
load_dotenv()

print("=" * 60)
print("🚀 Tourist Assistant Backend Server")
print("=" * 60)

# Load API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️  WARNING: GOOGLE_API_KEY not found")
if not GOOGLE_PLACES_API_KEY:
    print("⚠️  WARNING: GOOGLE_PLACES_API_KEY not found")

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global state
current_location = None
last_ai_response = None
last_identified_location = None
knowledge_base = []  # Stores PDF content
conversation_history = []

# --- Helper Functions ---

def call_gemini_api(prompt, system_message=None):
    """Call Google Gemini API directly"""
    if not GOOGLE_API_KEY:
        return "API key not configured"
    
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{url}?key={GOOGLE_API_KEY}",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            print(f"❌ Gemini API error: {response.status_code}")
            return f"API Error: {response.status_code}"
            
    except Exception as e:
        print(f"❌ Gemini API exception: {e}")
        return f"Error: {str(e)}"

def read_pdf(file_path):
    """Extract text from PDF"""
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""

def load_knowledge_base():
    """Load PDFs into knowledge base"""
    global knowledge_base
    
    try:
        kb_dir = "knowledge-base"
        os.makedirs(kb_dir, exist_ok=True)
        
        pdf_files = glob.glob(os.path.join(kb_dir, "*.pdf"))
        
        if not pdf_files:
            print("ℹ️  No PDFs in knowledge base")
            knowledge_base = []
            return "No PDFs found"
        
        print(f"📚 Loading {len(pdf_files)} PDF(s)...")
        
        knowledge_base = []
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            print(f"   📄 Reading: {filename}")
            content = read_pdf(pdf_file)
            
            if content:
                chunks = split_text(content)
                knowledge_base.append({
                    "filename": filename,
                    "content": content,
                    "chunks": chunks,
                    "size": os.path.getsize(pdf_file)
                })
        
        total_chunks = sum(len(doc["chunks"]) for doc in knowledge_base)
        print(f"✅ Loaded {len(knowledge_base)} file(s), {total_chunks} chunks")
        
        return f"Loaded {len(knowledge_base)} document(s) with {total_chunks} chunks"
        
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        return f"Error: {str(e)}"

def search_knowledge_base(query, top_k=3):
    """Simple keyword search in knowledge base"""
    if not knowledge_base:
        return []
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    for doc in knowledge_base:
        for i, chunk in enumerate(doc["chunks"]):
            chunk_lower = chunk.lower()
            # Simple scoring based on word matches
            matches = sum(1 for word in query_words if word in chunk_lower)
            if matches > 0:
                results.append({
                    "filename": doc["filename"],
                    "chunk": chunk,
                    "score": matches,
                    "chunk_index": i
                })
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def search_attractions(location):
    """Search Google Places API"""
    if not GOOGLE_PLACES_API_KEY:
        return {"error": "Places API not configured"}
    
    try:
        # Geocode
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        geocode_params = {"address": location, "key": GOOGLE_PLACES_API_KEY}
        geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
        geocode_data = geocode_response.json()
        
        if geocode_data["status"] != "OK":
            return {"error": f"Location not found: {location}"}
        
        location_data = geocode_data["results"][0]
        lat = location_data["geometry"]["location"]["lat"]
        lng = location_data["geometry"]["location"]["lng"]
        
        # Search places
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        places_params = {
            "location": f"{lat},{lng}",
            "radius": 5000,
            "type": "tourist_attraction",
            "key": GOOGLE_PLACES_API_KEY
        }
        places_response = requests.get(places_url, params=places_params, timeout=10)
        places_data = places_response.json()
        
        attractions = []
        for place in places_data.get("results", [])[:10]:
            attractions.append({
                "name": place["name"],
                "rating": place.get("rating", "N/A"),
                "vicinity": place.get("vicinity", "N/A")
            })
        
        return {
            "location": location_data["formatted_address"],
            "coordinates": {"lat": lat, "lng": lng},
            "attractions": attractions
        }
        
    except Exception as e:
        print(f"❌ Places API error: {e}")
        return {"error": str(e)}

def extract_location(message):
    """Extract location using Gemini"""
    prompt = (
        f"Extract the location or place name from this text. "
        f"Return ONLY the location name, nothing else. "
        f"If no location is mentioned, return 'None'.\n\n"
        f"Text: \"{message}\""
    )
    
    location = call_gemini_api(prompt)
    location = location.strip()
    
    if location.lower() in ["none", "not specified", "n/a", "no location"]:
        return None
    
    return location

def generate_chat_response(history, user_message):
    """Generate AI response"""
    global current_location, last_ai_response, last_identified_location
    
    system_message = (
        "You are an expert tourist assistant and travel guide. "
        "Provide helpful, accurate information about destinations, attractions, "
        "travel tips, and local culture. Keep responses concise (2-4 sentences) "
        "unless asked for details. Be friendly and enthusiastic."
    )
    
    # Build context from history
    context = ""
    for turn in history[-5:]:  # Last 5 turns
        if len(turn) >= 2:
            context += f"User: {turn[0]}\nAssistant: {turn[1]}\n\n"
    
    # Try to extract location
    location = extract_location(user_message)
    
    if not location and current_location:
        if any(kw in user_message.lower() for kw in ["attractions", "places", "visit", "see", "things to do"]):
            location = current_location
    
    # Search attractions if location found
    attraction_context = ""
    if location and GOOGLE_PLACES_API_KEY:
        attractions_data = search_attractions(location)
        if "error" not in attractions_data:
            last_identified_location = attractions_data["location"]
            attraction_context = f"\n\nLocation: {attractions_data['location']}\n"
            attraction_context += "Top Attractions:\n"
            for i, att in enumerate(attractions_data["attractions"][:5], 1):
                attraction_context += f"{i}. {att['name']} (Rating: {att['rating']})\n"
    
    # Search knowledge base
    kb_context = ""
    kb_results = search_knowledge_base(user_message)
    if kb_results:
        kb_context = "\n\nRelevant information from uploaded documents:\n"
        for result in kb_results:
            kb_context += f"\nFrom {result['filename']}:\n{result['chunk'][:300]}...\n"
    
    # Build final prompt
    full_prompt = f"{context}User: {user_message}{attraction_context}{kb_context}"
    
    # Generate response
    reply = call_gemini_api(full_prompt, system_message)
    last_ai_response = reply
    
    # Determine source
    source = "AI Assistant"
    if kb_results:
        source = "Knowledge Base + AI"
    elif attraction_context:
        source = "Places API + AI"
    
    return {
        "reply": reply,
        "source": source,
        "location_detected": last_identified_location,
        "kb_sources": [r["filename"] for r in kb_results] if kb_results else []
    }

# --- API Endpoints ---

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "google_api": GOOGLE_API_KEY is not None,
            "places_api": GOOGLE_PLACES_API_KEY is not None,
            "knowledge_base": len(knowledge_base) > 0
        },
        "documents_loaded": len(knowledge_base)
    })

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        history = data.get("history", [])
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"💬 User: {user_message}")
        response = generate_chat_response(history, user_message)
        print(f"🤖 Bot: {response['reply'][:100]}...")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice/transcribe", methods=["POST"])
def api_transcribe():
    """Speech to text - requires speech_recognition and pydub"""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    try:
        import speech_recognition as sr
        from pydub import AudioSegment
        
        audio_file = request.files["audio"]
        temp_wav = "temp_audio.wav"
        
        # Convert to WAV
        sound = AudioSegment.from_file(audio_file)
        sound.export(temp_wav, format="wav")
        
        # Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.remove(temp_wav)
        print(f"✅ Transcribed: {text}")
        
        return jsonify({"transcription": text, "success": True})
        
    except ImportError:
        return jsonify({"error": "Speech recognition not installed"}), 503
    except Exception as e:
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        print(f"❌ Transcription error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice/speak", methods=["POST"])
def api_text_to_speech():
    """Text to speech - requires gtts"""
    try:
        from gtts import gTTS
        
        data = request.json
        text = data.get("text", "").strip() or last_ai_response
        
        if not text:
            return jsonify({"error": "No text to speak"}), 400
        
        print(f"🔊 Generating speech: {text[:50]}...")
        
        tts = gTTS(text=text, lang="en", slow=False)
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        
        return send_file(
            audio_stream,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="speech.mp3"
        )
        
    except ImportError:
        return jsonify({"error": "TTS not installed"}), 503
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/location/set", methods=["POST"])
def api_set_location():
    global current_location, last_identified_location
    
    try:
        data = request.json
        location = data.get("location", "").strip()
        
        if not location:
            return jsonify({"error": "No location provided"}), 400
        
        print(f"📍 Setting location: {location}")
        
        attractions_data = search_attractions(location)
        
        if "error" in attractions_data:
            return jsonify({"error": attractions_data["error"]}), 404
        
        current_location = attractions_data["location"]
        last_identified_location = current_location
        
        print(f"✅ Location set: {current_location}")
        
        return jsonify({
            "success": True,
            "location": current_location,
            "coordinates": attractions_data.get("coordinates"),
            "attractions_found": len(attractions_data.get("attractions", []))
        })
        
    except Exception as e:
        print(f"❌ Set location error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/location/current", methods=["GET"])
def api_get_location():
    return jsonify({
        "current_location": current_location,
        "last_identified": last_identified_location
    })

@app.route("/api/documents/upload", methods=["POST"])
def api_upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400
    
    try:
        kb_dir = "knowledge-base"
        os.makedirs(kb_dir, exist_ok=True)
        
        filename = file.filename
        filepath = os.path.join(kb_dir, filename)
        
        if os.path.exists(filepath):
            return jsonify({"error": f"File '{filename}' already exists"}), 409
        
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        
        print(f"📄 Uploaded: {filename} ({file_size} bytes)")
        
        # Reload knowledge base
        status = load_knowledge_base()
        
        return jsonify({
            "success": True,
            "filename": filename,
            "size": file_size,
            "status": status
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents/list", methods=["GET"])
def api_list_documents():
    try:
        kb_dir = "knowledge-base"
        os.makedirs(kb_dir, exist_ok=True)
        
        pdf_files = glob.glob(os.path.join(kb_dir, "*.pdf"))
        
        documents = []
        for pdf_file in pdf_files:
            documents.append({
                "name": os.path.basename(pdf_file),
                "size": os.path.getsize(pdf_file),
                "modified": datetime.fromtimestamp(os.path.getmtime(pdf_file)).isoformat()
            })
        
        return jsonify({
            "documents": documents,
            "total": len(documents),
            "knowledge_base_active": len(knowledge_base) > 0
        })
        
    except Exception as e:
        print(f"❌ List error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents/delete/<filename>", methods=["DELETE"])
def api_delete_document(filename):
    try:
        kb_dir = "knowledge-base"
        filepath = os.path.join(kb_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        
        os.remove(filepath)
        print(f"🗑️  Deleted: {filename}")
        
        status = load_knowledge_base()
        
        return jsonify({
            "success": True,
            "filename": filename,
            "status": status
        })
        
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents/refresh", methods=["POST"])
def api_refresh_knowledge_base():
    try:
        status = load_knowledge_base()
        return jsonify({
            "success": True,
            "status": status,
            "files_loaded": len(knowledge_base)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Server Startup ---
if __name__ == "__main__":
    print("\n🔧 Initializing knowledge base...")
    load_knowledge_base()
    
    print("\n" + "=" * 60)
    print("✅ Server ready!")
    print("=" * 60)
    print("📡 API Endpoints:")
    print("   • Health:            GET  /api/health")
    print("   • Chat:              POST /api/chat")
    print("   • Voice Input:       POST /api/voice/transcribe")
    print("   • Voice Output:      POST /api/voice/speak")
    print("   • Set Location:      POST /api/location/set")
    print("   • Get Location:      GET  /api/location/current")
    print("   • Upload Document:   POST /api/documents/upload")
    print("   • List Documents:    GET  /api/documents/list")
    print("   • Delete Document:   DELETE /api/documents/delete/<name>")
    print("   • Refresh KB:        POST /api/documents/refresh")
    print("=" * 60)
    print(f"🌐 Server: http://localhost:8000")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)