# src/agent.py
import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq
from src.prompt import get_agent_prompt
from src.tools import (
    calculate_water_target,
    recipe_search_by_ingredient,
    filter_by_macros,
    generate_weekly_meal_plan,
    generate_shopping_list,
    calculate_bmi,
    nutrition_analysis
)

load_dotenv()

def build_nutrition_agent():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found! Check your .env file in the root folder.")

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )
    
    tools = [
        calculate_water_target,
        recipe_search_by_ingredient,
        filter_by_macros,
        generate_weekly_meal_plan,
        generate_shopping_list,
        calculate_bmi,
        nutrition_analysis
    ]
    
    prompt = get_agent_prompt()
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=12,
        handle_parsing_errors=True
    )