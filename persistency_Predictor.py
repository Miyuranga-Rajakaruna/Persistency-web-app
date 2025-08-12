
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify
import requests

# Load the updated Random Forest model
model = joblib.load('Model01.sav')

# Feature names in training order
feature_names = ['premium_amount','product_type',	'income_tier',	'billing_frequency','billing_channel','client_age','Agent_Vintage']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Customer_details.html')

@app.route('/getresults', methods=['POST'])
def getresults():
    result = request.form

    # Get input values from form
    premium_amount = float(result['premium_amount'])
    product_type = float(result['product_type'])
    income_tier = float(result['income_tier'])
    billing_frequency = float(result['billing_frequency'])
    billing_channel = float(result['billing_channel'])
    client_age = float(result['client_age'])
    agent_vintage = float(result['agent_vintage'])

    # Prepare input for prediction
    test_data = np.array([[premium_amount, product_type, income_tier, billing_frequency,billing_channel,client_age,agent_vintage]])
    
    # Make prediction
    prediction = model.predict(test_data)[0]
    probabilities = model.predict_proba(test_data)[0]

    # Feature importance analysis
  
    

    # Result dictionary for rendering
    resultDict = {
        "persistency": "Stay" if prediction == 0 else "Leave",
        "stay_prob": round(probabilities[0] * 100, 2),
        "leave_prob": round(probabilities[1] * 100, 2),
        
    }

    return render_template('persistency_results.html', results=resultDict)


# --- Hybrid Chatbot Logic ---
OPENROUTER_API_KEY = "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"  # Updated API key

# Extracted rule-based logic as a function
def get_rule_based_reply(user_input):
    if not user_input:
        return None
    if "score" in user_input:
        return "Your persistency score shows how likely a policyholder is to continue paying premiums. Higher scores indicate stronger customer commitment."
    elif "low score" in user_input:
        return "A low score means the customer is at risk of lapsing. Engage them immediately—understand their pain points, offer support, or revise their plan."
    elif "high score" in user_input:
        return "High scores suggest healthy persistency. Keep maintaining communication and offering proactive service to ensure it stays high."
    elif "improve" in user_input or "increase" in user_input:
        return "To improve persistency: train your agents well, communicate consistently, simplify processes, and follow up regularly before renewals."
    elif "why customer leave" in user_input or "reasons for lapse" in user_input:
        return "Top reasons include affordability, lack of understanding of policy benefits, poor service, or feeling disengaged. Identify the specific reason via feedback."
    elif "agent" in user_input:
        return "Agent vintage refers to their experience level. Senior agents tend to build better trust, explain policies clearly, and handle objections more effectively."
    elif "agent performance" in user_input:
        return "Track agent performance via metrics like first-year persistency, number of active follow-ups, conversion rates, and complaint levels."
    elif "crm" in user_input:
        return "CRM is critical. It helps track interactions, automate reminders, personalize messages, and segment customers for tailored service."
    elif "best time to contact" in user_input:
        return "Typically, 2–3 days before a due date or policy anniversary works best. Avoid weekends and aim for mid-morning calls."
    elif "digital" in user_input or "mobile app" in user_input:
        return "A mobile app helps customers view policies, make payments, and access documents anytime. It's essential for modern retention."
    elif "call center" in user_input or "support delay" in user_input:
        return "Minimize wait times, provide self-service options, and train staff for empathy. These factors directly affect customer retention."
    elif "personalized service" in user_input:
        return "Personalization builds loyalty. Use CRM data to address clients by name, refer to past conversations, and send relevant offers."
    elif "follow-up" in user_input:
        return "Follow-ups should be timely, professional, and empathetic. Don't just sell—check how the customer feels about their coverage."
    elif "complaint" in user_input or "angry customer" in user_input:
        return "Listen patiently. Acknowledge the issue. Provide clear steps for resolution. Turning a complaint into satisfaction increases loyalty."
    elif "customer segment" in user_input or "income level" in user_input:
        return "Segment customers by premium size, tenure, or policy type. Tailor service and communication style accordingly."
    elif "surrender" in user_input or "cancel policy" in user_input:
        return "Understand their reason. Sometimes offering a top-up benefit or switching to a smaller premium plan can prevent a surrender."
    elif "claim" in user_input or "settlement" in user_input:
        return "Fast, transparent claim settlement boosts trust. Guide them through documentation and timelines clearly."
    elif "retention strategy" in user_input:
        return "Retention improves with proactive service, timely engagement, personalized outreach, and feedback-driven improvements."
    elif "incentives" in user_input:
        return "Loyalty rewards, discounts on renewals, or even birthday greetings can enhance long-term engagement."
    elif "policy" in user_input:
        return "Policy is a contract between the insurer and the policyholder. It outlines the coverage and benefits provided by the insurer."
    elif "Who is owner of this webapp?" in user_input:
        return "This webapp is owned by Miyuranga Rajakaruna." 
    elif "Miyuranga Rajakaruna" in user_input:
        return "Miyuranga Rajakaruna Undergrad Smart Student at University of Moratuwa.His contact Details are 📱070-1070700/📧yasirumiyurangarajakaruna@gmail.com"
    elif "regulation" in user_input or "compliance" in user_input:
        return "Ensure all communications are in line with IRCSL and insurance board guidelines. Clear disclosure builds trust and reduces disputes."
    elif "what is persistency" in user_input:
        return "Persistency is the percentage of policies that stay in force after a specific period—e.g., 13-month, 25-month benchmarks. It’s a key health metric."
    elif "customer education" in user_input:
        return "Educating customers on benefits, policy terms, and claims process reduces lapses and dissatisfaction."
    elif "training" in user_input:
        return "Regular sales and service training for agents improves product understanding and client communication."
    elif "hello" in user_input or "hi" in user_input:
        return "Hi! I'm your assistant manager. Ask me anything about policy retention, scores, or improving customer experience."
    return None

def call_deepseek_api(user_input, api_key=OPENROUTER_API_KEY):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",  # Or your deployed site URL
        "X-Title": "AIA Persistency Chatbot"
    }
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {"role": "system", "content": (
                "You are an insurance persistency assistant. Always reply directly to the user in clear, actionable steps or concise explanations. "
                "Do not show your reasoning, internal thoughts, or planning. Only output the final answer as if you are speaking to the user. "
                "If you don't know the answer, say so politely."
            )},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"AI API status: {response.status_code}")
        print(f"AI API response: {response.text}")
        if response.status_code == 200:
            data = response.json()
            ai_reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # If content is blank, try to use 'reasoning' or any other available field
            if not ai_reply:
                # Try to extract 'reasoning' or any other non-empty field
                reasoning = data.get("choices", [{}])[0].get("message", {}).get("reasoning", "")
                if reasoning:
                    ai_reply = reasoning
            # Post-process: format numbered and bulleted lists as multi-line
            if ai_reply:
                import re
                def format_ai_reply(text):
                    import re
                    # Extract only the final answer after a quoted block or a phrase
                    match = re.search(r'(So[\s,]+we[\s,]+write:|Example of a direct reply:|"Certainly!.*)', text, re.IGNORECASE | re.DOTALL)
                    if match:
                        text = text[match.start():]
                    # Remove leading phrases like 'So we write:' or quotes
                    text = re.sub(r'^(So[\s,]+we[\s,]+write:|Example of a direct reply:|"|“|”)+', '', text, flags=re.IGNORECASE)
                    # Add line breaks before numbered items
                    text = re.sub(r'(\d+\.)', r'\n\1', text)
                    # Add line breaks after each numbered item
                    text = re.sub(r'(\d+\..*?)(?=\n\d+\.|$)', lambda m: m.group(1).strip() + '\n', text, flags=re.DOTALL)
                    # Remove Markdown bold/italic asterisks for cleaner look
                    text = re.sub(r'\*\*\s*', '', text)
                    text = re.sub(r'\n+', '\n', text)
                    text = re.sub(r' +\n', '\n', text)
                    text = re.sub(r'\n +', '\n', text)
                    return text.strip()
                ai_reply = format_ai_reply(ai_reply)
            return ai_reply if ai_reply else None
    except Exception as e:
        print(f"AI API error: {e}")
    return None

@app.route("/chat", methods=["POST"])
def hybrid_chat():
    user_input = None
    if request.is_json:
        data = request.get_json(silent=True)
        if data and 'message' in data:
            user_input = data['message'].lower()
    if not user_input:
        user_input = request.form.get('message', '').lower()

    if not user_input:
        return jsonify({"reply": "Sorry, I didn't get your message. Please try again."}), 400

    # 1. AI first
    ai_reply = call_deepseek_api(user_input)
    if ai_reply:
        return jsonify({"reply": ai_reply})

    # 2. Rule-based fallback
    rule_reply = get_rule_based_reply(user_input)
    if rule_reply:
        return jsonify({"reply": rule_reply})

    # 3. Fallback message
    fallback = "Sorry, I couldn’t understand that. You can ask me about policy retention, scores, or improving customer experience!"
    return jsonify({"reply": fallback})

if __name__ == '__main__':
    app.run(debug=True)
