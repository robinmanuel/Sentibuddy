from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import joblib
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/sentiment_model.pkl'
try:
    if os.path.exists(MODEL_PATH):
        sentiment_model = joblib.load(MODEL_PATH)
        print("Sentiment model loaded successfully")
    else:
        print(f"Model file not found at {MODEL_PATH}")
        sentiment_model = None
except Exception as e:
    print(f"Error loading sentiment model: {e}")
    sentiment_model = None

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    print("Gemini API initialized successfully")
except Exception as e:
    print(f"Error initializing Gemini API: {e}")
    model = None


def analyze_sentiment(text):
    if sentiment_model is None:
        text_lower = text.lower()
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        positive_keywords = ['happy', 'joy', 'glad', 'wonderful', 'excited', 'love', 'great']
        joyful_keywords = ['joyful', 'delighted', 'thrilled', 'blissful', 'ecstatic']
        content_keywords = ['content', 'satisfied', 'pleased', 'peaceful', 'calm']
        surprise_keywords = ['surprised', 'amazed', 'astonished', 'wow', 'unexpected']
        negative_keywords = ['sad', 'unhappy', 'disappointed', 'angry', 'upset', 'hate', 'terrible']
        anxiety_keywords = ['worried', 'nervous', 'anxious', 'stress', 'tension', 'concern']
        fear_keywords = ['afraid', 'scared', 'fearful', 'terrified', 'frightened']
        
        positive_score = sum(text_lower.count(word) for word in positive_keywords) 
        joyful_score = sum(text_lower.count(word) for word in joyful_keywords)
        content_score = sum(text_lower.count(word) for word in content_keywords)
        surprise_score = sum(text_lower.count(word) for word in surprise_keywords)
        negative_score = sum(text_lower.count(word) for word in negative_keywords)
        anxiety_score = sum(text_lower.count(word) for word in anxiety_keywords)
        fear_score = sum(text_lower.count(word) for word in fear_keywords)
        
        if positive_score > 0 or joyful_score > 0 or content_score > 0:
            if joyful_score > 0:
                return {"emotion": "joyful", "score": 0.8}
            elif exclamation_count >= 2:
                return {"emotion": "excited", "score": 0.8}
            elif content_score > 0:
                return {"emotion": "content", "score": 0.7}
            else:
                return {"emotion": "happy", "score": 0.7}
        elif surprise_score > 0 or (question_count > 0 and exclamation_count > 0):
            return {"emotion": "surprised", "score": 0.7}
        elif negative_score > 0:
            if anxiety_score > 0:
                return {"emotion": "anxious", "score": 0.75}
            elif fear_score > 0:
                return {"emotion": "fearful", "score": 0.7}
            else:
                if negative_score >= 3:
                    return {"emotion": "sad", "score": 0.7}
                else:
                    return {"emotion": "neutral", "score": 0.55}
        elif question_count > 0:
            return {"emotion": "curious", "score": 0.6}
        else:
            return {"emotion": "neutral", "score": 0.5}
    
    try:
        base_emotion = sentiment_model.predict([text])[0]
        
        if hasattr(sentiment_model, 'predict_proba'):
            probabilities = sentiment_model.predict_proba([text])[0]
            confidence = float(np.max(probabilities))
        else:
            decision_scores = sentiment_model.decision_function([text])
            raw_confidence = float(np.abs(decision_scores).mean())
            confidence = min(max(raw_confidence / 2.0, 0), 1)
        
        exclamation_count = text.count('!')
        question_count = text.count('?')
        has_caps = sum(1 for c in text if c.isupper()) / len(text) > 0.3
        
        positive_keywords = ['happy', 'joy', 'good', 'love', 'great', 'wonderful']
        joyful_keywords = ['joyful', 'delighted', 'thrilled', 'blissful', 'ecstatic']
        excitement_keywords = ['excited', 'wow', 'amazing', 'awesome', 'fantastic', 'pumped']
        content_keywords = ['content', 'satisfied', 'peaceful', 'calm', 'relaxed', 'serene']
        surprise_keywords = ['surprised', 'amazed', 'astonished', 'unexpected', 'shocked']
        curious_keywords = ['curious', 'interested', 'wonder', 'questioning', 'intrigued']
        negative_keywords = ['sad', 'bad', 'unhappy', 'disappointed', 'upset', 'down']
        anger_keywords = ['angry', 'mad', 'furious', 'rage', 'hate', 'outrage', 'annoyed']
        fear_keywords = ['afraid', 'scared', 'anxious', 'nervous', 'worry', 'frightened']
        anxious_keywords = ['anxious', 'anxiety', 'stressed', 'tense', 'uneasy', 'worried']
        
        text_lower = text.lower()
        positive_score = sum(2 if word in text_lower else 0 for word in positive_keywords)
        joyful_score = sum(2 if word in text_lower else 0 for word in joyful_keywords)
        excitement_score = sum(2 if word in text_lower else 0 for word in excitement_keywords)
        content_score = sum(2 if word in text_lower else 0 for word in content_keywords)
        surprise_score = sum(2 if word in text_lower else 0 for word in surprise_keywords)
        curious_score = sum(2 if word in text_lower else 0 for word in curious_keywords)
        negative_score = sum(2 if word in text_lower else 0 for word in negative_keywords)
        anger_score = sum(2 if word in text_lower else 0 for word in anger_keywords)
        fear_score = sum(2 if word in text_lower else 0 for word in fear_keywords)
        anxious_score = sum(2 if word in text_lower else 0 for word in anxious_keywords)
        
        emotion = base_emotion
        
        if base_emotion in ["happy", "joyful", "excited", "content"]:
            if joyful_score > 1 or (positive_score > 2 and exclamation_count >= 1):
                emotion = "joyful"
                confidence = max(confidence, 0.75)
            elif excitement_score > 0 or exclamation_count >= 2:
                emotion = "excited"
                confidence = max(confidence, 0.75)
            elif content_score > 0 or ("calm" in text_lower or "peace" in text_lower):
                emotion = "content"
                confidence = max(confidence, 0.7)
            elif surprise_score > 0:
                emotion = "surprised"
                confidence = max(confidence, 0.7)
            else:
                emotion = "happy"
                
        elif base_emotion in ["sad", "disappointed", "angry", "fearful", "anxious"]:
            if anger_score > 0 or (exclamation_count >= 2 and has_caps):
                emotion = "angry"
                confidence = max(confidence, 0.7)
            elif anxious_score > 1:
                emotion = "anxious"
                confidence = max(confidence, 0.7)
            elif fear_score > 0:
                emotion = "fearful"
                confidence = max(confidence, 0.65)
            elif "disappointed" in text_lower or "let down" in text_lower:
                emotion = "disappointed"
                confidence = max(confidence, 0.65)
            elif negative_score > 2:
                emotion = "sad"
                confidence = max(confidence, 0.6)
            else:
                if confidence < 0.65:
                    emotion = "neutral"
                    confidence = 0.55
        
        elif base_emotion in ["neutral", "surprised", "curious"]:
            if surprise_score > 0 or (exclamation_count > 0 and question_count > 0):
                emotion = "surprised"
                confidence = max(confidence, 0.65)
            elif curious_score > 0 or question_count > 1:
                emotion = "curious"
                confidence = max(confidence, 0.6)
            else:
                emotion = "neutral"
        
        return {
            "emotion": emotion,
            "score": confidence
        }
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return {"emotion": "neutral", "score": 0.5}

def generate_recommendation(emotion, text):
    if not model:
        return "I'd like to offer some support, but I'm having trouble connecting to my counseling system right now."
    
    prompt = f"""
    As a compassionate mental health professional, respond to this user message: "{text}"

    The user appears to be feeling {emotion}. 
    
    Create a thoughtful response that:
    1. Uses a warm, professional tone similar to an experienced therapist or counselor
    2. Validates their emotional experience without judgment
    3. Offers a gentle reframe or perspective that promotes emotional well-being
    4. Includes a specific, actionable suggestion tailored to their emotional state
    5. Provides authentic encouragement that respects their autonomy
    6. Recognizes that different emotions require different approaches:
       - For positive emotions: reinforce and build upon these feelings
       - For negative emotions: acknowledge pain while offering hope
       - For neutral emotions: promote mindfulness and self-awareness
    
    Keep your response under 100 words, conversational yet professional, and avoid clichés or overly simplistic advice.
    
    Respond directly without any preamble, roleplaying, or meta-commentary.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating recommendation: {e}")
        return "I recognize what you're experiencing. Sometimes our emotions give us important information about our needs. Consider taking a moment to check in with yourself about what might help right now. Remember that seeking support is a sign of strength, not weakness."

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        user_text = data['text']
        
        sentiment_result = analyze_sentiment(user_text)
        emotion = sentiment_result["emotion"]
        
        recommendation = generate_recommendation(emotion, user_text)
        
        return jsonify({
            "sentiment": emotion,
            "recommendation": recommendation,
            "confidence": float(sentiment_result["score"])
        })
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sentiment', methods=['POST'])
def sentiment_only():
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        user_text = data['text']
        
        sentiment_result = analyze_sentiment(user_text)
        
        return jsonify({
            "sentiment": sentiment_result["emotion"],
            "confidence": float(sentiment_result["score"])
        })
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        
        if not data or 'text' not in data or 'emotion' not in data:
            return jsonify({"error": "Both text and emotion must be provided"}), 400
        
        user_text = data['text']
        emotion = data['emotion']
        
        recommendation = generate_recommendation(emotion, user_text)
        
        return jsonify({
            "recommendation": recommendation
        })
        
    except Exception as e:
        print(f"Error generating recommendation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "apis": {
            "gemini": "available" if model else "unavailable"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)