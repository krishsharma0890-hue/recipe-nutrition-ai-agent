import os
import pandas as pd
from langchain.tools import tool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data", "recipes.csv")

DF = pd.read_csv(DATA_PATH)

def get_col(possible_names: list) -> str:
    """Helper function to find the exact column name regardless of casing/naming variations."""
    for col in DF.columns:
        if col.strip().lower() in [p.lower() for p in possible_names]:
            return col
    return possible_names[0]

# Automatically detect dataset column names dynamically
COL_NAME = get_col(['Name', 'recipe_name', 'title', 'RecipeName'])
COL_ING = get_col(['Ingredients', 'ingredients', 'recipe_ingredients', 'RecipeIngredients'])
COL_CAL = get_col(['Calories', 'calories', 'cal', 'CaloriesKcal'])
COL_PRO = get_col(['ProteinContent', 'protein', 'Protein', 'protein_g'])
COL_FAT = get_col(['FatContent', 'fat', 'Fat', 'fat_g'])
COL_CARB = get_col(['CarbohydrateContent', 'carbs', 'Carbohydrates', 'carbohydrates_g'])

@tool
def calculate_water_target(weight_kg: float) -> str:
    """Calculates recommended daily water intake based on weight in kilograms."""
    # Standard hydration formula: ~35ml per kg of body weight
    liters = (weight_kg * 35) / 1000
    return f"Recommended Daily Water Intake: {liters:.2f} Liters ({int(liters * 1000)} ml)"


@tool
def load_dataset_summary() -> str:
    """Returns a summary of the available recipe dataset including total count and column attributes."""
    return f"Dataset loaded with {len(DF)} recipes. Columns: {', '.join(DF.columns)}"


@tool
def recipe_search_by_ingredient(ingredients: str) -> str:
    """Searches for recipes containing specific comma-separated ingredients."""
    if COL_ING not in DF.columns:
        return f"Error: Ingredients column '{COL_ING}' not found in dataset."
        
    ing_list = [i.strip().lower() for i in ingredients.split(",")]
    
    matches = DF[DF[COL_ING].astype(str).str.lower().apply(lambda x: all(i in x for i in ing_list))]
    
    if matches.empty:
        return "No recipes found matching all specified ingredients."
        
    cols_to_show = [c for c in [COL_NAME, COL_CAL, COL_PRO, COL_ING] if c in DF.columns]
    results = matches[cols_to_show].head(5)
    return results.to_string(index=False)


@tool
def filter_by_macros(max_calories: float, min_protein: float) -> str:
    """Filters recipes by maximum calories and minimum protein in grams."""
    if COL_CAL not in DF.columns or COL_PRO not in DF.columns:
        return "Error: Required calorie or protein columns not found in dataset."

    filtered = DF[(DF[COL_CAL] <= max_calories) & (DF[COL_PRO] >= min_protein)]
    if filtered.empty:
        return f"No recipes found with <= {max_calories} cal and >= {min_protein}g protein."
        
    cols_to_show = [c for c in [COL_NAME, COL_CAL, COL_PRO, COL_FAT, COL_CARB] if c in DF.columns]
    return filtered[cols_to_show].head(5).to_string(index=False)


@tool
def nutrition_analysis(recipe_name: str) -> str:
    """Provides detailed nutritional breakdown (calories, protein, fat, carbs) for a given recipe name."""
    if COL_NAME not in DF.columns:
        return "Error: Recipe name column not found in dataset."

    match = DF[DF[COL_NAME].astype(str).str.lower() == recipe_name.lower()]
    if match.empty:
        return f"Recipe '{recipe_name}' not found."
        
    row = match.iloc[0]
    
    cal = row[COL_CAL] if COL_CAL in DF.columns else "N/A"
    pro = row[COL_PRO] if COL_PRO in DF.columns else "N/A"
    fat = row[COL_FAT] if COL_FAT in DF.columns else "N/A"
    carb = row[COL_CARB] if COL_CARB in DF.columns else "N/A"
    
    return f"Nutritional Info for {row[COL_NAME]}:\n- Calories: {cal} kcal\n- Protein: {pro}g\n- Fat: {fat}g\n- Carbs: {carb}g"


@tool
def generate_shopping_list(recipes_or_ingredients: str) -> str:
    """Generates a consolidated, categorized shopping list based on a list of recipes or ingredients provided as text."""
    if not recipes_or_ingredients:
        return "No ingredients provided to generate a shopping list."
        
    # Clean up and deduplicate ingredients
    items = [item.strip("- *").title() for item in recipes_or_ingredients.split("\n") if item.strip()]
    unique_items = sorted(list(set(items)))
    
    formatted_list = ["### 🛒 Consolidated Shopping List"]
    for item in unique_items:
        formatted_list.append(f"* {item}")
        
    return "\n".join(formatted_list)


@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """Calculates BMI and returns health category with recommended calorie targets."""
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)
    category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    return f"BMI: {bmi} ({category}). Recommended daily intake target: {'2200-2500' if category == 'Underweight' else '2000' if category == 'Normal' else '1500-1800'} kcal."


@tool
def generate_weekly_meal_plan(target_calories: float) -> str:
    """Generates a 7-day meal plan based on target daily calorie intake."""
    if COL_CAL not in DF.columns or COL_NAME not in DF.columns:
        return "Error: Required columns missing for meal plan generation."

    daily_target = target_calories / 3  # roughly split into 3 meals
    sample_meals = DF[DF[COL_CAL] <= daily_target]
    
    if len(sample_meals) < 7:
        return "Not enough low-calorie recipes in the dataset to generate a full weekly plan."
        
    sampled = sample_meals.sample(min(21, len(sample_meals)))
    
    plan = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_types = ["Breakfast", "Lunch", "Dinner"]
    
    idx = 0
    for day in days:
        plan.append(f"\n--- {day} ---")
        for m in meal_types:
            if idx < len(sampled):
                row = sampled.iloc[idx]
                pro_val = f", {row[COL_PRO]}g protein" if COL_PRO in DF.columns else ""
                plan.append(f"{m}: {row[COL_NAME]} ({row[COL_CAL]} kcal{pro_val})")
                idx += 1
    return "\n".join(plan)