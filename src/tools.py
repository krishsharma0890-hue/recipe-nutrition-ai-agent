"""Tools used by the nutrition assistant.

The bundled CSV is a recipe-review dataset, not a nutrition database.  These
tools therefore use the columns that actually exist and clearly say when a
nutrition value cannot be calculated from the available data.
"""

from pathlib import Path

import pandas as pd
from langchain.tools import tool

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "recipes.csv"
DF = pd.read_csv(DATA_PATH)

COL_NAME = "recipe_name" if "recipe_name" in DF.columns else None
COL_TEXT = "text" if "text" in DF.columns else None
COL_STARS = "stars" if "stars" in DF.columns else None
NUTRITION_COLUMNS = {"Calories", "ProteinContent", "FatContent", "CarbohydrateContent"}
HAS_NUTRITION_DATA = NUTRITION_COLUMNS.issubset(DF.columns)


def _require_recipe_names() -> str | None:
    if COL_NAME is None:
        return "The loaded dataset has no recipe_name column."
    return None


@tool
def calculate_water_target(weight_kg: float) -> str:
    """Calculate a general daily water target from body weight in kilograms."""
    if weight_kg <= 0:
        return "Weight must be greater than zero."
    liters = weight_kg * 0.035
    return f"General daily water target: {liters:.2f} L ({round(liters * 1000)} ml)."


@tool
def load_dataset_summary() -> str:
    """Describe the recipe dataset available to the assistant."""
    recipes = DF[COL_NAME].nunique() if COL_NAME else 0
    return (
        f"Dataset contains {len(DF)} reviews for {recipes} recipes. "
        f"Nutrition values available: {'yes' if HAS_NUTRITION_DATA else 'no'}. "
        f"Columns: {', '.join(DF.columns)}"
    )


@tool
def recipe_search_by_ingredient(ingredients: str) -> str:
    """Find recipe names whose titles or review text mention all comma-separated search terms."""
    error = _require_recipe_names()
    if error:
        return error
    terms = [term.strip().lower() for term in ingredients.split(",") if term.strip()]
    if not terms:
        return "Provide at least one ingredient or search term."

    searchable = DF[COL_NAME].fillna("").str.lower()
    if COL_TEXT:
        searchable = searchable + " " + DF[COL_TEXT].fillna("").str.lower()
    matches = DF[searchable.map(lambda value: all(term in value for term in terms))]
    if matches.empty:
        return "No recipes or reviews matched all specified terms."

    result = matches.groupby(COL_NAME, dropna=True).size().sort_values(ascending=False).head(5)
    return "Matching recipes (number of reviews):\n" + "\n".join(
        f"- {name}: {count}" for name, count in result.items()
    )


@tool
def filter_by_macros(max_calories: float, min_protein: float) -> str:
    """Filter recipes by calories and protein when the loaded dataset includes nutrition data."""
    if not HAS_NUTRITION_DATA:
        return "Cannot filter by macros: the bundled CSV contains recipe reviews, not calories or protein values."
    filtered = DF[(DF["Calories"] <= max_calories) & (DF["ProteinContent"] >= min_protein)]
    return filtered[[COL_NAME, "Calories", "ProteinContent", "FatContent", "CarbohydrateContent"]].head(5).to_string(index=False)


@tool
def nutrition_analysis(recipe_name: str) -> str:
    """Return nutrition values when present, otherwise summarize ratings and reviews for a recipe."""
    error = _require_recipe_names()
    if error:
        return error
    matches = DF[DF[COL_NAME].fillna("").str.casefold() == recipe_name.casefold()]
    if matches.empty:
        return f"Recipe '{recipe_name}' was not found."
    name = matches.iloc[0][COL_NAME]
    if HAS_NUTRITION_DATA:
        row = matches.iloc[0]
        return (f"Nutritional info for {name}:\n- Calories: {row['Calories']} kcal\n"
                f"- Protein: {row['ProteinContent']} g\n- Fat: {row['FatContent']} g\n"
                f"- Carbs: {row['CarbohydrateContent']} g")
    rating = f" Average rating: {matches[COL_STARS].mean():.1f}/5." if COL_STARS else ""
    return f"{name} has {len(matches)} review(s).{rating} Nutrition values are not available in this dataset."


@tool
def generate_shopping_list(recipes_or_ingredients: str) -> str:
    """Format newline- or comma-separated ingredients as a deduplicated shopping list."""
    items = [item.strip(" -•*\t").title() for item in recipes_or_ingredients.replace(",", "\n").splitlines()]
    items = sorted({item for item in items if item})
    if not items:
        return "No ingredients were provided."
    return "### Shopping List\n" + "\n".join(f"- {item}" for item in items)


@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """Calculate BMI from weight in kilograms and height in centimetres."""
    if weight_kg <= 0 or height_cm <= 0:
        return "Weight and height must be greater than zero."
    bmi = round(weight_kg / (height_cm / 100) ** 2, 1)
    category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    return f"BMI: {bmi} ({category}). This is a screening measure, not a diagnosis."


@tool
def generate_weekly_meal_plan(target_calories: float) -> str:
    """Generate a seven-day recipe rotation; calorie matching requires nutrition data."""
    error = _require_recipe_names()
    if error:
        return error
    names = DF[COL_NAME].dropna().drop_duplicates().head(7).tolist()
    if len(names) < 7:
        return "Not enough distinct recipe names to make a weekly rotation."
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    note = " Calorie totals cannot be verified because the dataset has no nutrition values." if not HAS_NUTRITION_DATA else ""
    return "Weekly recipe rotation:" + note + "\n" + "\n".join(
        f"- {day}: {name}" for day, name in zip(days, names)
    )
