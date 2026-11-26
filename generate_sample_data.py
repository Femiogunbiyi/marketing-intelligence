from pathlib import Path
import random
from typing import TypedDict

import pandas as pd


class Campaign(TypedDict):
    name: str
    channel: str
    roas_range: tuple[float, float]
    ctr_range: tuple[float, float]


class SampleRow(TypedDict):
    date: str


def create_lagos_ecommerce_sample(
    output_path: str = "data/raw/sample_lagos_ecommerce.csv",
    n_rows: int = 300,
    seed: int = 42,
) -> None:
    random.seed(seed)

    # Mix of high-ROAS winners and money-losing campaigns (to trigger real alerts!)
    campaigns: list[Campaign] = [
        {"name": "IG Skincare Retargeting", "channel": "Instagram", "roas_range": (3.2, 5.5), "ctr_range": (2.2, 4.0)},
        {"name": "Google Brand & Lagos Delivery", "channel": "Google Ads", "roas_range": (4.0, 6.8), "ctr_range": (4.5, 7.5)},
        {"name": "TikTok Influencer Broad", "channel": "TikTok", "roas_range": (0.5, 1.1), "ctr_range": (0.4, 0.9)},  # Triggers ROAS & CTR alert!
        {"name": "Meta Cold Traffic - Nationwide", "channel": "Facebook", "roas_range": (0.8, 1.3), "ctr_range": (0.8, 1.4)},  # Triggers ROAS alert!
        {"name": "WhatsApp VIP Broadcast", "channel": "WhatsApp", "roas_range": (5.0, 9.0), "ctr_range": (6.0, 11.0)},
    ]

    # Create 110 unique Nigerian customer emails (some repeat VIPs, some one-time buyers)
    first_names = ["chioma", "tunde", "amaka", "femi", "zainab", "chinedu", "adeola", "ibrahim", "ngozi", "segun", "kemi", "david"]
    last_names = ["okafor", "balogun", "adeyemi", "nwosu", "musa", "eze", "ogunbiyi", "suleiman", "mensah", "lawal"]
    customers = [
        f"{random.choice(first_names)}.{random.choice(last_names)}{i}@gmail.com"
        for i in range(1, 111)
    ]
    # Weight the first 25 customers so they act as repeat VIP buyers
    weights = [5 if i < 25 else 1 for i in range(len(customers))]

    dates = pd.date_range(start="2026-08-01", end="2026-11-20", freq="D")
    rows: list[dict[str, str | int | float]] = []

    for _ in range(n_rows):
        camp = random.choices(campaigns, weights=[30, 25, 20, 15, 10], k=1)[0]
        customer = random.choices(customers, weights=weights, k=1)[0]
        date_str = random.choice(dates).strftime("%Y-%m-%d")

        impressions = random.randint(2500, 25000)
        ctr = random.uniform(*camp["ctr_range"]) / 100
        clicks = max(1, int(impressions * ctr))

        spend_val = round(clicks * random.uniform(85, 220), 2)
        roas = random.uniform(*camp["roas_range"])
        revenue_val = round(spend_val * roas, 2)
        conversions = max(0, int(revenue_val // random.uniform(14000, 28000)))
        if conversions == 0:
            revenue_val = 0.0

        # Intentionally use raw Meta/Paystack column names & formatted Naira strings
        # to verify column_mapper.py and cleaning.py end-to-end!
        rows.append(
            {
                "Reporting Starts": date_str,
                "Campaign Name": camp["name"],
                "Platform": camp["channel"],
                "Customer Email": customer,
                "Impressions": impressions,
                "Link Clicks": clicks,
                "Purchases": conversions,
                "Amount Spent (NGN)": f"₦{spend_val:,.2f}",
                "Purchase Conversion Value": f"₦{revenue_val:,.2f}",
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Generated {len(rows)} rows at: {out}")


if __name__ == "__main__":
    create_lagos_ecommerce_sample()
