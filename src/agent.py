# src/agent.py

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq

from src.prompt import get_agent_prompt
from src.tools import (
    calculate_water_target,
    load_dataset_summary,
    recipe_search_by_ingredient,
    filter_by_macros,
    nutrition_analysis,
    get_complete_recipe_and_nutrition,
    generate_shopping_list,
    calculate_bmi,
    generate_weekly_meal_plan,
)

load_dotenv()


def build_nutrition_agent():

    # First try environment variables
    api_key = os.getenv("GROQ_API_KEY")

    # If running on Streamlit Cloud, try Streamlit Secrets
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file locally "
            "or Streamlit Cloud Secrets when deployed."
        )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=api_key,
    )

    tools = [
        calculate_water_target,
        load_dataset_summary,
        recipe_search_by_ingredient,
        filter_by_macros,
        nutrition_analysis,
        get_complete_recipe_and_nutrition,
        generate_shopping_list,
        calculate_bmi,
        generate_weekly_meal_plan,
    ]

    prompt = get_agent_prompt()

    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=12,
        handle_parsing_errors=True,
    )