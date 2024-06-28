from transformers import pipeline

# Load the NLP pipeline
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Sample product data
product_data = {
    "electronics": ["laptop", "smartphone", "headphones"],
    "clothing": ["shirt", "jeans", "jacket"],
    # Add more categories and products
}

# Sample intents and responses
intents = {
    "recommend_products": "Sure! Here are some products you might like: {}",
    "greet": "Hello! How can I assist you today?",
    "fallback": "I'm sorry, I didn't understand that. Could you please rephrase?"
}

def extract_intent_entities(text):
    entities = nlp(text)
    intent = None
    entity = None
    
    for ent in entities:
        if ent["entity"] in intents:
            intent = ent["entity"]
        elif ent["entity"] in product_data.keys():
            entity = ent["entity"]
    
    return intent, entity

def generate_response(intent, entity=None):
    if intent == "recommend_products":
        if entity in product_data:
            products = ", ".join(product_data[entity])
            return intents[intent].format(products)
        else:
            return "I'm sorry, I don't have recommendations for that category."
    elif intent in intents:
        return intents[intent]
    else:
        return intents["fallback"]

# Main interaction loop
def chat():
    print("Chatbot: " + generate_response("greet"))
    while True:
        user_input = input("You: ")
        intent, entity = extract_intent_entities(user_input)
        response = generate_response(intent, entity)
        print("Chatbot: " + response)

if __name__ == "__main__":
    chat()
