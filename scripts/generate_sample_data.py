from __future__ import annotations

import csv
import math
import random
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "naye_pankh_sample_impact_data.csv"


random.seed(42)

programs = [
    {
        "program": "Education Support",
        "campaign": "Back to School Drive",
        "base_donation": 52000,
        "base_beneficiaries": 165,
        "base_expense": 34000,
        "volunteer_factor": 1.20,
    },
    {
        "program": "Food Distribution",
        "campaign": "Meal Support Campaign",
        "base_donation": 43000,
        "base_beneficiaries": 260,
        "base_expense": 39000,
        "volunteer_factor": 1.05,
    },
    {
        "program": "Menstrual Hygiene",
        "campaign": "Pad for Dignity Drive",
        "base_donation": 38000,
        "base_beneficiaries": 195,
        "base_expense": 22000,
        "volunteer_factor": 0.92,
    },
    {
        "program": "Clothing Support",
        "campaign": "Warmth and Dignity Drive",
        "base_donation": 35000,
        "base_beneficiaries": 180,
        "base_expense": 26000,
        "volunteer_factor": 1.00,
    },
    {
        "program": "Health Awareness",
        "campaign": "Community Wellness Camp",
        "base_donation": 36000,
        "base_beneficiaries": 210,
        "base_expense": 25000,
        "volunteer_factor": 0.95,
    },
]

cities = [
    {"city": "Kanpur", "state": "Uttar Pradesh", "scale": 1.22},
    {"city": "Delhi", "state": "Delhi", "scale": 1.16},
    {"city": "Lucknow", "state": "Uttar Pradesh", "scale": 0.98},
    {"city": "Jaipur", "state": "Rajasthan", "scale": 0.90},
    {"city": "Patna", "state": "Bihar", "scale": 0.84},
    {"city": "Bhopal", "state": "Madhya Pradesh", "scale": 0.78},
]

donor_mix = {
    "Individual": 0.48,
    "Corporate CSR": 0.24,
    "Community Group": 0.18,
    "Alumni Network": 0.10,
}

channels = ["UPI", "Bank Transfer", "Fundraising Event", "Online Portal"]


def weighted_choice(weights: dict[str, float]) -> str:
    marker = random.random()
    cumulative = 0.0
    for label, weight in weights.items():
        cumulative += weight
        if marker <= cumulative:
            return label
    return next(reversed(weights))


def month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def seasonal_multiplier(month: int) -> float:
    festive_boost = 0.18 if month in {8, 9, 10, 11, 12} else 0
    summer_slowdown = -0.08 if month in {4, 5, 6} else 0
    school_cycle = 0.12 if month in {6, 7} else 0
    wave = 0.06 * math.sin(month / 12 * math.tau)
    return 1 + festive_boost + summer_slowdown + school_cycle + wave


def donor_amount_multiplier(donor_type: str) -> float:
    return {
        "Individual": 0.86,
        "Corporate CSR": 1.45,
        "Community Group": 1.05,
        "Alumni Network": 0.92,
    }[donor_type]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    months = [
        month_start(year, month)
        for year in (2024, 2025)
        for month in range(1, 13)
    ]

    for month_index, month_date in enumerate(months, start=1):
        long_term_growth = 1 + (month_index - 1) * 0.012
        season = seasonal_multiplier(month_date.month)
        is_festive_season = int(month_date.month in {8, 9, 10, 11, 12})

        for city_info in cities:
            for program_info in programs:
                donor_type = weighted_choice(donor_mix)
                channel = random.choice(channels)
                city_scale = city_info["scale"]
                program_noise = random.uniform(0.88, 1.14)
                impact_noise = random.uniform(0.90, 1.18)

                donation = (
                    program_info["base_donation"]
                    * city_scale
                    * season
                    * long_term_growth
                    * donor_amount_multiplier(donor_type)
                    * program_noise
                )
                expense = (
                    program_info["base_expense"]
                    * city_scale
                    * (0.96 + season / 16)
                    * random.uniform(0.91, 1.10)
                )
                beneficiaries = (
                    program_info["base_beneficiaries"]
                    * city_scale
                    * impact_noise
                    * (1 + (month_index - 1) * 0.006)
                )
                volunteer_hours = (
                    beneficiaries
                    * 0.30
                    * program_info["volunteer_factor"]
                    * random.uniform(0.86, 1.20)
                )
                event_count = max(1, round(beneficiaries / random.uniform(70, 115)))
                new_donors = max(3, round(donation / random.uniform(3900, 6200)))
                recurring_donors = max(1, round(new_donors * random.uniform(0.28, 0.52)))
                social_reach = round((beneficiaries * random.uniform(16, 32)) + (donation / 12))
                engagement_rate = round(random.uniform(3.2, 9.8), 2)
                cost_per_beneficiary = expense / beneficiaries
                quarter = f"Q{(month_date.month - 1) // 3 + 1}"

                rows.append(
                    {
                        "date": month_date.isoformat(),
                        "year": month_date.year,
                        "month_num": month_date.month,
                        "month": month_date.strftime("%b"),
                        "quarter": quarter,
                        "is_festive_season": is_festive_season,
                        "city": city_info["city"],
                        "state": city_info["state"],
                        "program": program_info["program"],
                        "campaign": program_info["campaign"],
                        "donor_type": donor_type,
                        "channel": channel,
                        "donation_amount": round(donation),
                        "expense_amount": round(expense),
                        "beneficiaries_reached": round(beneficiaries),
                        "volunteer_hours": round(volunteer_hours, 1),
                        "event_count": event_count,
                        "new_donors": new_donors,
                        "recurring_donors": recurring_donors,
                        "social_reach": social_reach,
                        "engagement_rate": engagement_rate,
                        "cost_per_beneficiary": round(cost_per_beneficiary, 2),
                    }
                )

    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
