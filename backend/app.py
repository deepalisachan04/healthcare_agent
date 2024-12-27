import os
import logging
import uuid
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langsmith import Client, traceable
from dotenv import load_dotenv
from flask_cors import CORS
from langchain.memory import ConversationBufferMemory

# Step 1: Load API keys from .env file
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

if not google_api_key or not langsmith_api_key:
    raise ValueError("API keys are missing. Please set GOOGLE_API_KEY and LANGSMITH_API_KEY in the .env file.")

# Step 2: Initialize LangSmith client (Only once)
client = Client(api_key=langsmith_api_key)

logging.basicConfig(level=logging.DEBUG)

# Step 3: Set up the Google Generative AI model
model_name = "gemini-1.5-flash"
temperature = 0.7

llm = ChatGoogleGenerativeAI(
    google_api_key=google_api_key,
    model=model_name,
    temperature=temperature
)

# Step 4: Define the prompt templates
mental_health_prompt = PromptTemplate(
    input_variables=["user_name", "query"],
    template=(
        "Provide a thoughtful, informative, and supportive response to the following query from {user_name}: {query}. "
        "While offering medically relevant advice where appropriate, ensure to highlight that it is not a substitute for professional healthcare. "
        "Suggest helpful resources, coping strategies, and recommend consulting a healthcare provider for personalized care."
    )
)

health_report_prompt = PromptTemplate(
    input_variables=["user_name", "age", "gender", "medical_history", "current_medications"],
    template=(
        "Generate a health report summary for the following patient details:\n" 
        "Name: {user_name}\nAge: {age}\nGender: {gender}\nMedical History: {medical_history}\nCurrent Medications: {current_medications}.\n" 
        "Provide a detailed summary, including potential health considerations and recommendations for further care."
    )
)

# Step 5: MentalHealthAgent class
class MentalHealthAgent:
    def __init__(self):
        self.client = client
        self.llm = llm
        self.memory = {}

    @traceable(client=client)
    def answer_user_query(self, user_id, user_name, query):
        if user_id not in self.memory:
            self.memory[user_id] = ConversationBufferMemory(memory_key="conversation", return_messages=True)

        prompt = mental_health_prompt.format(user_name=user_name, query=query)
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])

        if isinstance(response, AIMessage):
            response_content = response.content.strip()
            self.memory[user_id].save_context({"input": message.content}, {"output": response_content})
            response_content += " 💖 Feel free to ask me anything, I'm here for you!"
            return response_content
        else:
            return "Oops, something went wrong. Let me try again. 💔"

    def generate_health_report(self, user_name, age, gender, medical_history, current_medications):
        prompt = health_report_prompt.format(
            user_name=user_name, age=age, gender=gender,
            medical_history=medical_history, current_medications=current_medications
        )
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])

        if isinstance(response, AIMessage):
            return response.content.strip()
        else:
            return "Failed to generate the health report. Please try again."

    def clear_conversation(self, user_id):
        if user_id in self.memory:
            del self.memory[user_id]

# Global instance of MentalHealthAgent
mental_health_agent = MentalHealthAgent()

# Step 6: Initialize Flask API
app = Flask(__name__)

# Add CORS for all routes and methods
CORS(app, resources={r"/*": {"origins": "*"}})

# Step 7: Endpoints

# Handle mental health queries
@app.route('/ask_mental_health_agent', methods=['POST'])
def ask_mental_health_agent():
    data = request.json
    user_id = data.get('user_id', str(uuid.uuid4()))
    user_name = data['user_name']
    query = data['query']

    response = mental_health_agent.answer_user_query(user_id, user_name, query)
    return jsonify({"response": response})

# Generate health report
@app.route('/generate_health_report', methods=['POST'])
def generate_health_report():
    data = request.json
    user_name = data['user_name']
    age = data['age']
    gender = data['gender']
    medical_history = data['medical_history']
    current_medications = data['current_medications']

    report = mental_health_agent.generate_health_report(user_name, age, gender, medical_history, current_medications)
    return jsonify({"report": report})

# Clear conversation log
@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    data = request.json
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    mental_health_agent.clear_conversation(user_id)
    return jsonify({"message": "Conversation log cleared successfully."})

# Step 8: Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
