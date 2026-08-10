import re
import os
import streamlit as st
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

# MUST be called before importing/initializing ChatGroq
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from src.agent import build_nutrition_agent
from src.tools import get_complete_recipe_and_nutrition, generate_shopping_list

st.set_page_config(page_title="Recipe & Nutrition AI Agent", page_icon="🥗", layout="wide")

st.title("🥗 Recipe & Nutrition AI Agent")
st.write("Your personal AI assistant for diet planning, recipe recommendations, and macro tracking.")

# Robust regex parser matching any variation of Protein, Carbs, Fat in LLM text
def extract_macros(text: str):
    protein_match = re.search(r'\*{0,2}Protein\*{0,2}\s*:?\s*~?(\d+(?:\.\d+)?)\s*g', text, re.IGNORECASE)
    carbs_match = re.search(r'\*{0,2}(?:Carbs?|Carbohydrates)\*{0,2}\s*:?\s*~?(\d+(?:\.\d+)?)\s*g', text, re.IGNORECASE)
    fat_match = re.search(r'\*{0,2}Fats?\*{0,2}\s*:?\s*~?(\d+(?:\.\d+)?)\s*g', text, re.IGNORECASE)

    if protein_match and carbs_match and fat_match:
        return {
            "Protein": float(protein_match.group(1)),
            "Carbs": float(carbs_match.group(1)),
            "Fat": float(fat_match.group(1))
        }
    return None

# Helper parser to extract weight and height from user messages
def extract_measurements(text: str):
    weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|kilos|kilograms)', text, re.IGNORECASE)
    height_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeters|meters)', text, re.IGNORECASE)
    
    measurements = {}
    if weight_match:
        measurements["weight"] = float(weight_match.group(1))
    if height_match:
        val = float(height_match.group(1))
        if val < 3.0:
            val = val * 100
        measurements["height"] = val
    return measurements

# Cache agent startup
@st.cache_resource
def get_nutrition_agent():
    return build_nutrition_agent()

# Initialize session state widget keys if they don't exist yet
if "weight_input" not in st.session_state:
    st.session_state.weight_input = 70.0

if "height_input" not in st.session_state:
    st.session_state.height_input = 170.0

if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_macros" not in st.session_state:
    st.session_state.current_macros = {
        "Nutrient": ["Protein", "Carbs", "Fat"],
        "Grams": [30, 50, 15]
    }

if "chart_title" not in st.session_state:
    st.session_state.chart_title = "Sample Meal Breakdown"

# PRE-PROCESS: Extract measurements BEFORE rendering sidebar widgets
if st.session_state.pending_user_input:
    parsed = extract_measurements(st.session_state.pending_user_input)
    if "weight" in parsed:
        st.session_state.weight_input = parsed["weight"]
    if "height" in parsed:
        st.session_state.height_input = parsed["height"]

# Sidebar
with st.sidebar:
    st.header("⚡ Quick Utilities")
    st.subheader("BMI Calculator")
    
    weight = st.number_input(
        "Weight (kg)", 
        min_value=30.0, 
        max_value=200.0, 
        step=0.5,
        key="weight_input"
    )
    height = st.number_input(
        "Height (cm)", 
        min_value=100.0, 
        max_value=220.0, 
        step=1.0,
        key="height_input"
    )

    # Automatically compute BMI on state change
    if height > 0:
        bmi = weight / ((height / 100) ** 2)
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25.0:
            category = "Normal"
        elif 25.0 <= bmi < 30.0:
            category = "Overweight"
        else:
            category = "Obese"
            
        st.metric("Your BMI", f"{bmi:.1f}", delta=category, delta_color="off")

    st.divider()

    # =========================================================================
    # 🍕 MACRO PIE CHART SECTION START
    # =========================================================================
    st.subheader("Macro Distribution Visualizer")
    
    macro_df = pd.DataFrame(st.session_state.current_macros)
    fig = px.pie(
        macro_df, 
        values="Grams", 
        names="Nutrient", 
        title=st.session_state.chart_title,
        color="Nutrient",
        color_discrete_map={"Protein": "#1f77b4", "Carbs": "#17becf", "Fat": "#ff7f0e"}
    )
    st.plotly_chart(fig, use_container_width=True)
    # =========================================================================
    # 🍕 MACRO PIE CHART SECTION END
    # =========================================================================

try:
    agent_executor = get_nutrition_agent()
except ValueError as error:
    agent_executor = None
    st.warning(f"AI assistant unavailable: {error}")

# Render historic messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process input
if user_input := st.chat_input("Ask for recipes, weekly plans, or macro advice..."):
    # Check if user mentioned height/weight measurements
    parsed_measurements = extract_measurements(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing and querying tools..."):
            if agent_executor is None:
                output = "Add a valid `GROQ_API_KEY` to `.env`, then restart the app to enable AI responses."
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
                st.stop()
            
            # SMART INTERCEPTOR: Differentiate shopping lists vs recipes/meal ideas cleanly
            query_lower = user_input.lower()
            if "shopping list" in query_lower and "recipe" not in query_lower and "breakfast" not in query_lower:
                output = generate_shopping_list.invoke({"dish_name": user_input})
            elif any(kw in query_lower for kw in ["recipe", "breakfast", "lunch", "dinner", "ilish", "chicken", "paneer", "curry", "pasta", "salad", "soup", "biryani", "fish"]):
                output = get_complete_recipe_and_nutrition.invoke({"dish_name": user_input})
            else:
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                output = response["output"]

            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})

            # Check output for macros
            extracted = extract_macros(output)
            if extracted:
                st.session_state.current_macros = {
                    "Nutrient": ["Protein", "Carbs", "Fat"],
                    "Grams": [extracted["Protein"], extracted["Carbs"], extracted["Fat"]]
                }
                st.session_state.chart_title = "Latest Recipe Macro Breakdown"

            # If measurements were extracted, queue them and trigger rerun
            if parsed_measurements:
                st.session_state.pending_user_input = user_input
                st.rerun()
            elif extracted:
                st.rerun()