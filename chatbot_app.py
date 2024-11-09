import streamlit as st
import joblib
import spacy
import re
from pymongo import MongoClient

# Load spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Load your classifier and vectorizer
classifier = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["procurement_db"]
collection = db["purchase_orders"]

# Function to extract entities from the input
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {"YEAR": None, "DEPARTMENT": None, "ITEM": None, "QUARTER": None, "ACQUISITION_METHOD": None}

    # Extract entities using spaCy
    for ent in doc.ents:
        if ent.label_ == "DATE" and re.fullmatch(r"\d{4}", ent.text):
            entities["YEAR"] = int(ent.text)
        elif ent.label_ == "ORG":
            entities["DEPARTMENT"] = ent.text.strip()

    # Regex-based extraction for years
    year_match = re.search(r"\b(20\d{2})\b", user_input)
    if year_match:
        entities["YEAR"] = int(year_match.group(1))

    # Extract quarter
    if "Q1" in user_input or "quarter 1" in user_input:
        entities["QUARTER"] = "Q1"
    elif "Q2" in user_input or "quarter 2" in user_input:
        entities["QUARTER"] = "Q2"
    elif "Q3" in user_input or "quarter 3" in user_input:
        entities["QUARTER"] = "Q3"
    elif "Q4" in user_input or "quarter 4" in user_input:
        entities["QUARTER"] = "Q4"

    # Check MongoDB for matching department, item, and acquisition method names
    departments = collection.distinct("Department Name")
    for dept in departments:
        if dept.lower() in user_input.lower():
            entities["DEPARTMENT"] = dept
            break

    items = collection.distinct("Item Name")
    for item in items:
        if item.lower() in user_input.lower():
            entities["ITEM"] = item
            break

    acquisition_methods = collection.distinct("Acquisition Method")
    for method in acquisition_methods:
        if method.lower() in user_input.lower():
            entities["ACQUISITION_METHOD"] = method
            break

    return entities

# Function to generate responses based on intent and entities
def get_response(predicted_intent, entities):
    # Handle various intents based on prediction
    if predicted_intent == "total_spending":
        match = {}
        if entities["DEPARTMENT"]:
            match["Department Name"] = {"$regex": entities["DEPARTMENT"], "$options": "i"}
        if entities["YEAR"]:
            match["Purchase Year"] = entities["YEAR"]
        if entities["QUARTER"]:
            match["Purchase Quarter"] = entities["QUARTER"]

        pipeline = [
            {"$match": match},
            {"$group": {"_id": None, "total_spending": {"$sum": "$Total Price"}}}
        ]
        result = list(collection.aggregate(pipeline))
        if result and "total_spending" in result[0]:
            total_spending = result[0]["total_spending"]
            return f"The total spending is ${total_spending:,.2f}."
        else:
            return "No data available for the specified query."

    elif predicted_intent == "orders_count":
        match = {}
        if entities["YEAR"]:
            match["Purchase Year"] = entities["YEAR"]
        if entities["QUARTER"]:
            match["Purchase Quarter"] = entities["QUARTER"]

        order_count = collection.count_documents(match)
        return f"The total number of orders is {order_count}."

    elif predicted_intent == "frequent_items":
        pipeline = [
            {"$group": {"_id": "$Item Name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        result = list(collection.aggregate(pipeline))
        items = ", ".join([f"{item['_id']} ({item['count']} times)" for item in result])
        return f"The most frequently ordered items are: {items}."

    elif predicted_intent == "supplier_information":
        if entities["ITEM"]:
            result = collection.find_one({"Item Name": {"$regex": entities["ITEM"], "$options": "i"}}, {"Supplier Name": 1})
            if result:
                return f"The supplier for {entities['ITEM']} is {result['Supplier Name']}."
            return f"No supplier information found for {entities['ITEM']}."
        else:
            return "Please specify an item."

    elif predicted_intent == "item_price_information":
        if entities["ITEM"]:
            result = collection.find_one({"Item Name": {"$regex": entities["ITEM"], "$options": "i"}}, {"Unit Price": 1})
            if result:
                return f"The unit price for {entities['ITEM']} is ${result['Unit Price']:.2f}."
            return f"No pricing information found for {entities['ITEM']}."
        else:
            return "Please specify an item."

    return "I'm not sure how to answer that."

# Streamlit user interface setup
st.title("Procurement Chatbot")
user_input = st.text_input("You: ", placeholder="Ask a question here")

if user_input:
    # Predict intent
    input_features = vectorizer.transform([user_input])
    predicted_intent = classifier.predict(input_features)[0]

    # Extract entities
    entities = extract_entities(user_input)

    # Generate and display response
    response = get_response(predicted_intent, entities)
    st.write(f"**Chatbot:** {response}")
