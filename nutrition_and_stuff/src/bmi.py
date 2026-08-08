def calculate_bmi_value(weight_kg: float, height_cm: float) -> dict:
    if height_cm <= 0:
        return {"error": "Height must be greater than 0."}
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)
    category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    return {"bmi": bmi, "category": category}