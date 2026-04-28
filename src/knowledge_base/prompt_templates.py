"""Gemini prompt template strings for all 10 KB summary types.

Each template instructs Gemini to produce summaries that are:

    - Human and conversational in tone, as if explaining to a stakeholder
    - Factually grounded — all numbers come from the statistical input
    - 3-5 sentences in length — informative but compact
    - Free from speculation — no inferred causes or trends

Templates deliberately avoid academic phrasing in favour of plain English
to produce summaries that read naturally when retrieved by the RAG pipeline
and presented to a stakeholder-facing system.

Templates use Python str.format() placeholders. The matching prompt
builders in prompt_builders.py supply the values.
"""


# ─────────────────────────────────────────────────────────────────────────────
# GEFCom Templates (5)
# ─────────────────────────────────────────────────────────────────────────────

GEFCOM_DAILY_TEMPLATE = """\
You are helping build a knowledge base for an energy demand forecasting thesis.
Write a short, natural paragraph (3-5 sentences) that a utility manager could
read and immediately understand. Avoid academic phrasing — write as if you are
briefly explaining the day to a colleague. Use the exact numbers provided.
Do not speculate about causes or compare to other periods.

Data for Zone {zone_id} on {date} ({dow}):
  Average hourly load : {load_mean:.1f} MW
  Lowest hourly load  : {load_min:.1f} MW
  Highest hourly load : {load_max:.1f} MW
  Variability (std)   : {load_std:.1f} MW
  Total daily energy  : {load_sum:.1f} MWh
  Hours with data     : {obs_count}/24

Write the paragraph now:"""


GEFCOM_WEEKLY_TEMPLATE = """\
You are helping build a knowledge base for an energy demand forecasting thesis.
Write a short, natural paragraph (3-5 sentences) that a utility manager could
read and quickly understand. Write conversationally — as if summarising the week
to a colleague in a brief meeting. Use the exact numbers provided.
Do not speculate about causes or compare to other weeks.

Data for Zone {zone_id}, ISO Week {iso_week} of {iso_year}:
  Average of daily means : {weekly_mean:.1f} MW
  Lowest daily mean      : {weekly_min:.1f} MW
  Highest daily mean     : {weekly_max:.1f} MW
  Day-to-day variability : {weekly_std:.1f} MW

Write the paragraph now:"""


GEFCOM_MONTHLY_TEMPLATE = """\
You are helping build a knowledge base for an energy demand forecasting thesis.
Write a short, natural paragraph (3-5 sentences) a utility planner could read
and immediately understand. Keep the language plain and direct — as if you are
giving a brief verbal update. Use the exact numbers provided.
Do not speculate about external causes.

Data for Zone {zone_id}, {month_name} {year}:
  Monthly mean of daily means : {monthly_mean:.1f} MW
  Lowest daily mean            : {monthly_min:.1f} MW
  Highest daily mean           : {monthly_max:.1f} MW
  Day-to-day variability       : {monthly_std:.1f} MW

Write the paragraph now:"""


GEFCOM_SEASONAL_TEMPLATE = """\
You are helping build a knowledge base for an energy demand forecasting thesis.
Write a short, natural paragraph (3-5 sentences) that reads like a plain-English
seasonal briefing — the kind a utility manager would share in a planning meeting.
Use the exact numbers provided. Mention the season and what the load levels mean
in practical terms, but do not speculate about specific external causes.

Data for Zone {zone_id}, {season} {year}:
  Seasonal mean of daily means : {seasonal_mean:.1f} MW
  Lowest daily mean             : {seasonal_min:.1f} MW
  Highest daily mean            : {seasonal_max:.1f} MW
  Day-to-day variability        : {seasonal_std:.1f} MW
  Days with data                : {day_count}

Write the paragraph now:"""


GEFCOM_SYSTEM_LEVEL_TEMPLATE = """\
You are helping build a knowledge base for an energy demand forecasting thesis.
Write a short, natural paragraph (3-5 sentences) that a grid operator could read
and understand at a glance. Write as if briefing a colleague on the system-wide
picture — keep it plain and direct. Always make clear this is the system-level
total (the sum of all 20 zones, equivalent to Zone 21 in the original GEFCom
competition). Use the exact numbers provided.

System-level data ({granularity}) for {date_label}:
  Total system mean load   : {load_mean:.1f} MW
  Total system minimum     : {load_min:.1f} MW
  Total system maximum     : {load_max:.1f} MW
  System variability (std) : {load_std:.1f} MW
  Total system energy      : {load_sum:.1f} MWh

Write the paragraph now:"""


# ─────────────────────────────────────────────────────────────────────────────
# Household Templates (5)
# ─────────────────────────────────────────────────────────────────────────────

HOUSEHOLD_DAILY_TEMPLATE = """\
You are helping build a knowledge base for a household energy consumption thesis.
Write a short, natural paragraph (3-5 sentences) that reads like a friendly
daily energy summary — the kind an energy advisor might send to a homeowner.
Keep it warm and accessible, not technical. Use the exact numbers provided.
Do not speculate about behaviour or causes beyond what the numbers show.

Household data for {date}:
  Average active power    : {gap_mean:.3f} kW
  Lowest active power     : {gap_min:.3f} kW
  Highest active power    : {gap_max:.3f} kW
  Average voltage         : {volt_mean:.1f} V
  Average current         : {gi_mean:.2f} A
  Kitchen appliances      : {sm1_mean:.2f} Wh average
  Laundry appliances      : {sm2_mean:.2f} Wh average
  Water heater / HVAC     : {sm3_mean:.2f} Wh average

Write the paragraph now:"""


HOUSEHOLD_WEEKLY_TEMPLATE = """\
You are helping build a knowledge base for a household energy consumption thesis.
Write a short, natural paragraph (3-5 sentences) that reads like a weekly energy
summary for a homeowner — conversational, easy to understand, and focused on
what the numbers actually mean for day-to-day living. Use the exact numbers.
Do not speculate about causes.

Household data for the week ending {period_start}:
  Average active power  : {gap_mean:.3f} kW
  Lowest active power   : {gap_min:.3f} kW
  Highest active power  : {gap_max:.3f} kW
  Kitchen appliances    : {sm1_mean:.2f} Wh average
  Laundry appliances    : {sm2_mean:.2f} Wh average
  Water heater / HVAC   : {sm3_mean:.2f} Wh average

Write the paragraph now:"""


HOUSEHOLD_MONTHLY_TEMPLATE = """\
You are helping build a knowledge base for a household energy consumption thesis.
Write a short, natural paragraph (3-5 sentences) that reads like a monthly energy
report summary — plain-English, accessible to a non-technical homeowner, and
focused on what the figures tell us about how the household used energy that month.
Use the exact numbers provided. Do not speculate about causes.

Household data for {month_year}:
  Average active power  : {gap_mean:.3f} kW
  Lowest active power   : {gap_min:.3f} kW
  Highest active power  : {gap_max:.3f} kW
  Month-to-month spread : {gap_std:.3f} kW
  Kitchen appliances    : {sm1_mean:.2f} Wh average
  Laundry appliances    : {sm2_mean:.2f} Wh average
  Water heater / HVAC   : {sm3_mean:.2f} Wh average

Write the paragraph now:"""


HOUSEHOLD_APPLIANCE_TEMPLATE = """\
You are helping build a knowledge base for a household energy consumption thesis.
Write a short, natural paragraph (3-5 sentences) that clearly explains how the
household's energy was divided between its main appliance groups during this period.
Write as if explaining to a homeowner who wants to understand their energy bill.
Name all three appliance groups (kitchen, laundry, water heater/HVAC).
Use the exact numbers and percentages provided.

Appliance data for {date_label}:
  Kitchen appliances (Sub-metering 1):
    Average {sm1_mean:.2f} Wh  |  Range: {sm1_min:.2f}-{sm1_max:.2f} Wh  |  Share: {sm1_share:.1f}%
  Laundry appliances (Sub-metering 2):
    Average {sm2_mean:.2f} Wh  |  Range: {sm2_min:.2f}-{sm2_max:.2f} Wh  |  Share: {sm2_share:.1f}%
  Water heater and HVAC (Sub-metering 3):
    Average {sm3_mean:.2f} Wh  |  Range: {sm3_min:.2f}-{sm3_max:.2f} Wh  |  Share: {sm3_share:.1f}%
  Total sub-metering average : {total_sm_mean:.2f} Wh
  Total household active power: {gap_mean:.3f} kW

Write the paragraph now:"""


HOUSEHOLD_YEARLY_TEMPLATE = """\
You are helping build a knowledge base for a household energy consumption thesis.
Write a short, natural paragraph (3-5 sentences) that gives a clear annual
picture of the household's energy use — conversational, as if summarising the
year for the household in a year-end energy report. Name the peak season and
the dominant appliance group. Use the exact numbers provided.

Annual household data for {year}:
  Average monthly power     : {yearly_mean:.3f} kW
  Lowest monthly average    : {yearly_min:.3f} kW
  Highest monthly average   : {yearly_max:.3f} kW
  Month-to-month spread     : {yearly_std:.3f} kW
  Peak season               : {peak_season}
  Kitchen appliances (annual mean)  : {sm1_mean:.2f} Wh
  Laundry appliances (annual mean)  : {sm2_mean:.2f} Wh
  Water heater / HVAC (annual mean) : {sm3_mean:.2f} Wh

Write the paragraph now:"""
