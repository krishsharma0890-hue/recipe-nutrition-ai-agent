import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RECIPES_CSV = os.path.join(DATA_DIR, "recipes.csv")

# Storage Paths
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
PROFILES_DIR = os.path.join(USER_DATA_DIR, "profiles")
PROGRESS_DIR = os.path.join(USER_DATA_DIR, "progress")

GENERATED_DIR = os.path.join(BASE_DIR, "generated")
MEAL_PLANS_DIR = os.path.join(GENERATED_DIR, "meal_plans")
REPORTS_DIR = os.path.join(GENERATED_DIR, "reports")