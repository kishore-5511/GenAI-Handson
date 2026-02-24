import os
from groq import Groq
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Use a current, supported model
CURRENT_MODEL = "llama-3.3-70b-versatile"

# 2. Define Experts
MODEL_CONFIG = {
    "technical": {
        "system_prompt": "You are a Senior Software Engineer. Provide concise, code-heavy solutions and debug errors with technical precision.",
        "model": CURRENT_MODEL
    },
    "billing": {
        "system_prompt": "You are a Billing Specialist. Be empathetic, focus on financial security, and explain refund/subscription policies clearly.",
        "model": CURRENT_MODEL
    },
    "general": {
        "system_prompt": "You are a helpful Customer Assistant. Provide friendly, general information for non-specific inquiries.",
        "model": CURRENT_MODEL
    }
}

# 3. The Router (The Core Task)
def route_prompt(user_input):
    # Defining the variable inside the function scope
    router_prompt = f"""
    Classify the following user query into exactly ONE of these categories: 
    [technical, billing, general].
    
    Return ONLY the category name. Do not include punctuation or explanations.
    
    Query: "{user_input}"
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": router_prompt}],
        model=CURRENT_MODEL,
        temperature=0, 
    )
    
    return response.choices[0].message.content.strip().lower()

# 4. The Orchestrator
def process_request(user_input):
    # Step A: Route
    category = route_prompt(user_input)
    print(f"--- [Router Log]: Routing to {category.upper()} expert ---")
    
    # Step B: Select Config
    config = MODEL_CONFIG.get(category, MODEL_CONFIG["general"])
    
    # Step C: Generate Expert Response
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": user_input}
        ],
        model=config["model"],
        temperature=0.7, 
    )
    
    return response.choices[0].message.content

# --- Testing ---
print(process_request("My python script is throwing an IndexError on line 5."))
print("\n" + "="*50 + "\n")
print(process_request("I was charged twice for my subscription this month."))