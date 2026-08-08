def calculate_daily_water_intake(weight_kg: float, activity_min: int = 0) -> float:
    base_liters = (weight_kg * 35) / 1000
    extra_liters = (activity_min / 30) * 0.35
    return round(base_liters + extra_liters, 2)