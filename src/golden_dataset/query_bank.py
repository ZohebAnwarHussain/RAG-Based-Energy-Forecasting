"""Query template bank for golden dataset generation.

50 queries total across three sources:

    GEFCOM_QUERIES      20 queries — GEFCom-specific
    HOUSEHOLD_QUERIES   18 queries — Household-specific
    CROSS_SCALE_QUERIES 12 queries — Cross-scale (both datasets)

Each query dict contains:
    user_query                  Natural language question
    query_type                  One of 7 types (see below)
    difficulty_level            easy / medium / hard
    query_scope                 daily / weekly / monthly / etc.
    granularity_target          Granularity for context selection
    retrieval_strategy_target   dense / hybrid / hierarchical / all
    answer_must_include         Required terms for correct answers
    answer_must_not_include     Forbidden terms (hallucination guard)
    retrieval_notes             Guidance for scoring retrieval quality
    evaluation_notes            Edge case guidance for RAGAS scoring

Query types:
    statistical    Exact numeric facts retrieval
    pattern        Temporal or behavioural pattern retrieval
    comparative    Multi-chunk synthesis and comparison
    zone_specific  Zone-based chunking (hybrid retrieval)
    appliance      Sub-metering retrieval (hybrid retrieval)
    operational    Actionable stakeholder insights
    cross_scale    Joint retrieval across both datasets
"""

from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# GEFCom Queries (20)
# ─────────────────────────────────────────────────────────────────────────────

GEFCOM_QUERIES: List[Dict[str, Any]] = [

    # ── Statistical (5) ───────────────────────────────────────────────────────
    {
        "user_query": "What was the average daily electricity load and peak load recorded across GEFCom zones during the dataset period?",
        "query_type": "statistical", "difficulty_level": "easy",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["MW", "average", "load"],
        "answer_must_not_include": ["Sub_metering", "voltage", "household"],
        "retrieval_notes": "Should retrieve daily summary rows containing load_mean and load_max values.",
        "evaluation_notes": "Accept if MW values and daily averages are referenced.",
    },
    {
        "user_query": "What is the typical range of minimum and maximum hourly load observed in a single day across the GEFCom zones?",
        "query_type": "statistical", "difficulty_level": "easy",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["MW", "minimum", "maximum"],
        "answer_must_not_include": ["household", "Sub_metering", "watt"],
        "retrieval_notes": "Tests retrieval of load_min and load_max from daily summaries.",
        "evaluation_notes": "Answer must reference both min and max values in MW.",
    },
    {
        "user_query": "Which months show the highest average electricity demand across GEFCom zones based on historical data?",
        "query_type": "statistical", "difficulty_level": "medium",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["month", "demand", "MW"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Should retrieve monthly aggregate summaries and compare monthly_mean values.",
        "evaluation_notes": "Accept if specific months are named with supporting MW evidence.",
    },
    {
        "user_query": "What is the standard deviation of daily load across GEFCom zones and what does it indicate about demand variability?",
        "query_type": "statistical", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["standard deviation", "variability", "MW"],
        "answer_must_not_include": ["household", "appliance"],
        "retrieval_notes": "Tests whether load_std values are retrieved and correctly interpreted.",
        "evaluation_notes": "Answer must interpret variability meaning, not just state the number.",
    },
    {
        "user_query": "What is the total daily energy consumption in MWh typically observed across GEFCom zones on weekdays versus weekends?",
        "query_type": "statistical", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["MWh", "weekday", "weekend"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Keyword 'weekday/weekend' + semantic meaning — tests hybrid retrieval.",
        "evaluation_notes": "Answer must distinguish weekday from weekend demand with MWh values.",
    },

    # ── Pattern (5) ───────────────────────────────────────────────────────────
    {
        "user_query": "Describe the typical daily electricity load pattern observed across GEFCom zones including morning rise, afternoon stability and evening peak behaviour.",
        "query_type": "pattern", "difficulty_level": "easy",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["peak", "morning", "evening", "load"],
        "answer_must_not_include": ["household", "Sub_metering", "voltage"],
        "retrieval_notes": "Core intraday pattern query.",
        "evaluation_notes": "Accept if at least two intraday phases are described.",
    },
    {
        "user_query": "How does electricity demand change between winter and summer months in the GEFCom dataset?",
        "query_type": "pattern", "difficulty_level": "medium",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["winter", "summer", "seasonal", "demand"],
        "answer_must_not_include": ["household", "appliance", "Sub_metering"],
        "retrieval_notes": "Should retrieve monthly summaries from both winter and summer months.",
        "evaluation_notes": "Answer must compare both seasons with directional claim.",
    },
    {
        "user_query": "What weekly demand patterns are observed in the GEFCom data? Are there consistent weekday versus weekend differences?",
        "query_type": "pattern", "difficulty_level": "medium",
        "query_scope": "weekly", "granularity_target": "weekly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["weekday", "weekend", "weekly", "demand"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Requires weekly parent + daily child context.",
        "evaluation_notes": "Answer must reference both weekly summary and daily-level variation.",
    },
    {
        "user_query": "Are there periods of unusually high demand variability in the GEFCom zones and what might explain them?",
        "query_type": "pattern", "difficulty_level": "hard",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["variability", "demand", "zone"],
        "answer_must_not_include": ["household", "Sub_metering", "voltage"],
        "retrieval_notes": "Requires retrieving high load_std rows.",
        "evaluation_notes": "Mark as hallucination if causes not grounded in retrieved summaries.",
    },
    {
        "user_query": "How does the weekly mean load evolve across ISO weeks in the GEFCom dataset? Describe any observable trend.",
        "query_type": "pattern", "difficulty_level": "hard",
        "query_scope": "weekly", "granularity_target": "weekly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["weekly", "trend", "mean", "load"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Tests hierarchical parent-level retrieval depth.",
        "evaluation_notes": "Accept if trend direction stated with week-level evidence.",
    },

    # ── Comparative (4) ───────────────────────────────────────────────────────
    {
        "user_query": "Compare electricity demand levels between high-demand months and low-demand months in the GEFCom dataset.",
        "query_type": "comparative", "difficulty_level": "medium",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["high", "low", "month", "demand", "MW"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Multi-chunk comparison.",
        "evaluation_notes": "Answer must name specific contrasting months with MW values.",
    },
    {
        "user_query": "How does daily load variability differ between high-demand periods and low-demand periods in the GEFCom dataset?",
        "query_type": "comparative", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["variability", "high-demand", "low-demand"],
        "answer_must_not_include": ["household", "appliance"],
        "retrieval_notes": "Tests synthesis of multiple retrieved chunks.",
        "evaluation_notes": "Answer must compare standard deviations across demand periods.",
    },
    {
        "user_query": "Compare weekly demand patterns in early versus late weeks of the year in the GEFCom dataset.",
        "query_type": "comparative", "difficulty_level": "hard",
        "query_scope": "weekly", "granularity_target": "weekly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["early", "late", "weekly", "demand"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Temporal comparison across ISO weeks.",
        "evaluation_notes": "Reject if ISO week numbers not grounded in retrieved context.",
    },
    {
        "user_query": "How does seasonal demand variability in GEFCom compare between the Spring and Autumn transition seasons?",
        "query_type": "comparative", "difficulty_level": "hard",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["Spring", "Autumn", "seasonal", "demand"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Tests seasonal summary retrieval for transition seasons.",
        "evaluation_notes": "Accept if both seasons compared with MW-level evidence.",
    },

    # ── Zone-specific (3) ─────────────────────────────────────────────────────
    {
        "user_query": "What are the distinctive load characteristics of the system-level zone (Zone 21) compared to individual zones in GEFCom?",
        "query_type": "zone_specific", "difficulty_level": "hard",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["Zone 21", "system", "load"],
        "answer_must_not_include": ["household", "Sub_metering", "appliance"],
        "retrieval_notes": "Zone 21 is synthetic system-level sum of all 20 zones.",
        "evaluation_notes": "Penalise if Zone 21 system-level nature not mentioned.",
    },
    {
        "user_query": "Describe the daily demand behaviour of a specific GEFCom zone and explain how it differs from the system-level average.",
        "query_type": "zone_specific", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["zone", "daily", "demand", "average"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Tests zone-level chunking. Hybrid needed for zone_id filtering.",
        "evaluation_notes": "Accept if a specific zone referenced with MW values.",
    },
    {
        "user_query": "Which GEFCom zones exhibit the highest seasonal demand variation between winter and summer?",
        "query_type": "zone_specific", "difficulty_level": "hard",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["zone", "winter", "summer", "seasonal"],
        "answer_must_not_include": ["household", "Sub_metering"],
        "retrieval_notes": "Zone + season combined query.",
        "evaluation_notes": "Answer must name zone IDs with seasonal MW evidence.",
    },

    # ── Operational (3) ───────────────────────────────────────────────────────
    {
        "user_query": "Based on historical GEFCom demand patterns, during which periods should grid operators prepare for peak load conditions?",
        "query_type": "operational", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["peak", "operator", "demand", "period"],
        "answer_must_not_include": ["household", "Sub_metering", "I think", "probably"],
        "retrieval_notes": "Stakeholder-facing operational query.",
        "evaluation_notes": "Mark as hallucination if not grounded in retrieved summaries.",
    },
    {
        "user_query": "What seasonal patterns in the GEFCom data should inform annual capacity planning decisions for utility operators?",
        "query_type": "operational", "difficulty_level": "hard",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["seasonal", "capacity", "planning", "demand"],
        "answer_must_not_include": ["household", "Sub_metering", "I think", "probably"],
        "retrieval_notes": "Tests multi-chunk synthesis of seasonal monthly summaries.",
        "evaluation_notes": "Primary hallucination test for GEFCom.",
    },
    {
        "user_query": "What does GEFCom weekly load data suggest about the best time windows for scheduled grid maintenance activities?",
        "query_type": "operational", "difficulty_level": "medium",
        "query_scope": "weekly", "granularity_target": "weekly",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["maintenance", "weekly", "load", "demand"],
        "answer_must_not_include": ["household", "Sub_metering", "speculate"],
        "retrieval_notes": "Operational planning query.",
        "evaluation_notes": "Reject if maintenance windows suggested without weekly load evidence.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Household Queries (18)
# ─────────────────────────────────────────────────────────────────────────────

HOUSEHOLD_QUERIES: List[Dict[str, Any]] = [

    # ── Statistical (4) ───────────────────────────────────────────────────────
    {
        "user_query": "What is the average global active power consumption of the household and how does it vary across different time periods?",
        "query_type": "statistical", "difficulty_level": "easy",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["kW", "average", "consumption"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Tests retrieval of Global_active_power_mean from daily summaries.",
        "evaluation_notes": "Accept if kW values and time period context are both referenced.",
    },
    {
        "user_query": "What are the typical voltage and current intensity levels in the household and do they remain stable over time?",
        "query_type": "statistical", "difficulty_level": "easy",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["voltage", "current", "intensity"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Tests retrieval of Voltage_mean and Global_intensity_mean.",
        "evaluation_notes": "Accept if Voltage (~230V) and intensity (A) values mentioned.",
    },
    {
        "user_query": "What is the average monthly energy consumption of the household and which months show the highest power usage?",
        "query_type": "statistical", "difficulty_level": "medium",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["monthly", "kW", "consumption"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Tests hierarchical retrieval of monthly parent summaries.",
        "evaluation_notes": "Answer must name specific high-consumption months.",
    },
    {
        "user_query": "How variable is the household daily power consumption? What does the standard deviation of global active power indicate?",
        "query_type": "statistical", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["standard deviation", "variability", "power"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Tests interpretation of Global_active_power_std.",
        "evaluation_notes": "Answer must interpret variability meaning, not just cite numbers.",
    },

    # ── Appliance (4) ─────────────────────────────────────────────────────────
    {
        "user_query": "Which sub-metering channel (kitchen, laundry, or HVAC and water heater) contributes the most to household electricity consumption on average?",
        "query_type": "appliance", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["sub-metering", "kitchen", "laundry", "HVAC"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "'Sub_metering' keyword critical for hybrid retrieval.",
        "evaluation_notes": "Answer must rank or compare all three sub-meters.",
    },
    {
        "user_query": "How does kitchen appliance usage (Sub-metering 1) vary across different days of the week based on household data?",
        "query_type": "appliance", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["kitchen", "Sub_metering_1", "daily"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Tests keyword retrieval for 'Sub_metering_1'.",
        "evaluation_notes": "Accept if weekday/weekend variation described with Wh values.",
    },
    {
        "user_query": "Describe the typical contribution of the HVAC and water heating sub-meter (Sub-metering 3) to total household consumption across seasons.",
        "query_type": "appliance", "difficulty_level": "hard",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["Sub_metering_3", "HVAC", "seasonal"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Requires both keyword match (Sub_metering_3) and seasonal context.",
        "evaluation_notes": "Reject if seasonal HVAC claim not supported by monthly summaries.",
    },
    {
        "user_query": "How does laundry appliance usage (Sub-metering 2) change between weekdays and weekends in the household dataset?",
        "query_type": "appliance", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["Sub_metering_2", "laundry", "weekday", "weekend"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Tests hybrid retrieval combining 'Sub_metering_2' and 'weekend' keywords.",
        "evaluation_notes": "Answer must compare weekday vs weekend laundry usage with Wh values.",
    },

    # ── Pattern (4) ───────────────────────────────────────────────────────────
    {
        "user_query": "Describe the typical daily electricity consumption pattern of the household including morning, afternoon and evening behaviour.",
        "query_type": "pattern", "difficulty_level": "easy",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["morning", "evening", "consumption", "kW"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Core daily pattern query.",
        "evaluation_notes": "Accept if at least two intraday phases described.",
    },
    {
        "user_query": "How does household electricity consumption change across seasons? Which season shows the highest consumption?",
        "query_type": "pattern", "difficulty_level": "medium",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["seasonal", "winter", "summer", "consumption"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Requires monthly parent summaries for full context.",
        "evaluation_notes": "Answer must name the highest season with monthly evidence.",
    },
    {
        "user_query": "Are there weekly consumption patterns visible in the household data? How does weekday consumption compare to weekend consumption?",
        "query_type": "pattern", "difficulty_level": "medium",
        "query_scope": "weekly", "granularity_target": "weekly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["weekday", "weekend", "weekly", "consumption"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Requires weekly parent + daily child context.",
        "evaluation_notes": "Answer must reference both weekly summary and daily-level breakdown.",
    },
    {
        "user_query": "How has the household annual energy consumption trended across the years covered by the dataset?",
        "query_type": "pattern", "difficulty_level": "hard",
        "query_scope": "multi_granularity", "granularity_target": "mixed",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["annual", "trend", "year", "consumption"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Requires yearly summary retrieval across multiple years.",
        "evaluation_notes": "Accept if year-over-year trend described with kW evidence.",
    },

    # ── Comparative (3) ───────────────────────────────────────────────────────
    {
        "user_query": "Compare household electricity consumption during high-demand months versus low-demand months. What drives the difference?",
        "query_type": "comparative", "difficulty_level": "medium",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["high", "low", "month", "consumption", "kW"],
        "answer_must_not_include": ["zone", "GEFCom"],
        "retrieval_notes": "Multi-chunk comparative query.",
        "evaluation_notes": "Answer must name contrasting months with kW values.",
    },
    {
        "user_query": "How does the ratio of sub-metering consumption to total household consumption change across different time periods?",
        "query_type": "comparative", "difficulty_level": "hard",
        "query_scope": "multi_granularity", "granularity_target": "mixed",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["sub-metering", "total", "ratio", "consumption"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Tests synthesis of sub-metering and total consumption values.",
        "evaluation_notes": "Hard query — accept partial answer if sub-metering and total compared.",
    },
    {
        "user_query": "Compare the household energy consumption profile in the first year of the dataset versus the final year. Has usage changed?",
        "query_type": "comparative", "difficulty_level": "hard",
        "query_scope": "multi_granularity", "granularity_target": "mixed",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["year", "consumption", "change", "kW"],
        "answer_must_not_include": ["zone", "MW", "GEFCom"],
        "retrieval_notes": "Requires yearly summaries from multiple years.",
        "evaluation_notes": "Reject if year-specific kW values not grounded in yearly summaries.",
    },

    # ── Operational (3) ───────────────────────────────────────────────────────
    {
        "user_query": "Based on historical household consumption patterns, during which hours and days should a household consider shifting flexible loads to reduce peak demand?",
        "query_type": "operational", "difficulty_level": "medium",
        "query_scope": "daily", "granularity_target": "daily",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["peak", "shift", "consumption", "demand"],
        "answer_must_not_include": ["zone", "MW", "GEFCom", "speculate"],
        "retrieval_notes": "Demand-side management query.",
        "evaluation_notes": "Mark as hallucination if not grounded in retrieved summaries.",
    },
    {
        "user_query": "What household consumption evidence suggests the best opportunity for installing a home energy management system to reduce costs?",
        "query_type": "operational", "difficulty_level": "hard",
        "query_scope": "multi_granularity", "granularity_target": "mixed",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["consumption", "peak", "energy", "management"],
        "answer_must_not_include": ["zone", "MW", "GEFCom", "I think", "probably"],
        "retrieval_notes": "High-level operational insight. Primary hallucination test.",
        "evaluation_notes": "Flag any unsupported claim.",
    },
    {
        "user_query": "What does the HVAC and water heater sub-metering data suggest about the best tariff plan for this household?",
        "query_type": "operational", "difficulty_level": "hard",
        "query_scope": "monthly", "granularity_target": "monthly",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["HVAC", "tariff", "consumption", "Sub_metering_3"],
        "answer_must_not_include": ["zone", "MW", "GEFCom", "I think"],
        "retrieval_notes": "Appliance-informed operational query.",
        "evaluation_notes": "Accept only if HVAC patterns support the tariff suggestion.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Scale Queries (12)
# ─────────────────────────────────────────────────────────────────────────────

CROSS_SCALE_QUERIES: List[Dict[str, Any]] = [

    # ── Daily cross-scale (3) ─────────────────────────────────────────────────
    {
        "user_query": "How do household-level electricity consumption patterns compare to zone-level demand patterns in terms of daily peak timing and magnitude?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["household", "zone", "peak", "daily"],
        "answer_must_not_include": ["speculate", "I think", "probably"],
        "retrieval_notes": "Requires retrieval from BOTH GEFCom and household daily KB chunks.",
        "evaluation_notes": "Answer must reference both kW (household) and MW (GEFCom) scales.",
    },
    {
        "user_query": "Do GEFCom zone-level daily load profiles show similar morning and evening peak shapes to household-level consumption profiles?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "daily",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["morning", "evening", "peak", "household", "zone"],
        "answer_must_not_include": ["speculate", "I think"],
        "retrieval_notes": "Intraday shape comparison across scales.",
        "evaluation_notes": "Answer must draw from both daily GEFCom and household summaries.",
    },
    {
        "user_query": "How does the daily demand variability at the household level compare to the daily load variability observed across GEFCom zones?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "daily",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["variability", "household", "zone", "daily"],
        "answer_must_not_include": ["speculate", "I think"],
        "retrieval_notes": "Variability comparison across scales.",
        "evaluation_notes": "Answer must compare std values from both kW and MW scales.",
    },

    # ── Weekly cross-scale (3) ────────────────────────────────────────────────
    {
        "user_query": "Do household-level weekly consumption patterns mirror the weekday-weekend differences observed in the multi-zone GEFCom utility data?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "weekly",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["weekday", "weekend", "household", "zone"],
        "answer_must_not_include": ["speculate", "I think"],
        "retrieval_notes": "Keywords 'weekday/weekend' needed alongside cross-dataset retrieval.",
        "evaluation_notes": "Answer must confirm or deny the mirror pattern with evidence from both.",
    },
    {
        "user_query": "How does the week-to-week variability in household energy use compare to the week-to-week variability in GEFCom zone demand?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "weekly",
        "retrieval_strategy_target": "dense",
        "answer_must_include": ["weekly", "variability", "household", "zone"],
        "answer_must_not_include": ["speculate", "I think"],
        "retrieval_notes": "Cross-scale weekly variability comparison.",
        "evaluation_notes": "Answer must reference weekly_std values from both datasets.",
    },
    {
        "user_query": "What can weekly patterns in both household and GEFCom data reveal about the typical energy demand cycle across different scales?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "weekly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["weekly", "cycle", "household", "zone", "demand"],
        "answer_must_not_include": ["speculate", "I think", "probably"],
        "retrieval_notes": "Synthesis query. Tests hierarchical retrieval from both datasets.",
        "evaluation_notes": "Answer must synthesise weekly patterns from both sources.",
    },

    # ── Monthly / Seasonal cross-scale (3) ────────────────────────────────────
    {
        "user_query": "Compare the seasonal demand variation observed at the household level with seasonal load variation in the GEFCom multi-zone dataset.",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "monthly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["seasonal", "household", "zone", "winter", "summer"],
        "answer_must_not_include": ["speculate", "I think"],
        "retrieval_notes": "Tests hierarchical retrieval of monthly summaries from both datasets.",
        "evaluation_notes": "Answer must compare seasonal patterns from both with evidence from each.",
    },
    {
        "user_query": "Which months show the greatest alignment between household-level consumption peaks and GEFCom zone-level demand peaks?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "monthly",
        "retrieval_strategy_target": "hybrid",
        "answer_must_include": ["month", "peak", "household", "zone", "alignment"],
        "answer_must_not_include": ["speculate", "I think"],
        "retrieval_notes": "Month-level cross-scale alignment query.",
        "evaluation_notes": "Answer must name specific months with evidence from both datasets.",
    },
    {
        "user_query": "How does the peak season identified in household yearly consumption data align with the peak seasonal demand periods in the GEFCom zone-level data?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "monthly",
        "retrieval_strategy_target": "hierarchical",
        "answer_must_include": ["peak season", "household", "zone", "seasonal", "demand"],
        "answer_must_not_include": ["speculate", "I think", "probably"],
        "retrieval_notes": "Requires yearly household summary (peak_season) and GEFCom seasonal summary.",
        "evaluation_notes": "Answer must reference peak season from both datasets with evidence.",
    },

    # ── Synthesis cross-scale (3) ─────────────────────────────────────────────
    {
        "user_query": "What insights can be drawn from combining household appliance-level sub-metering data with multi-zone utility load data for energy planning purposes?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "mixed",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["sub-metering", "zone", "planning", "energy"],
        "answer_must_not_include": ["speculate", "I think", "probably"],
        "retrieval_notes": "Core thesis-contribution test query.",
        "evaluation_notes": "Answer must draw evidence from both datasets. Primary hallucination test.",
    },
    {
        "user_query": "How can understanding fine-grained household consumption behaviour improve the interpretation of broader zone-level demand forecasting insights?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "mixed",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["household", "zone", "forecasting", "demand"],
        "answer_must_not_include": ["speculate", "I think", "probably"],
        "retrieval_notes": "High-level synthesis. Answer must connect household and GEFCom evidence.",
        "evaluation_notes": "Any claim not traceable to retrieved chunks should be flagged.",
    },
    {
        "user_query": "What does a combined analysis of household-level appliance patterns and zone-level load data reveal about the drivers of peak electricity demand?",
        "query_type": "cross_scale", "difficulty_level": "hard",
        "query_scope": "cross_scale", "granularity_target": "mixed",
        "retrieval_strategy_target": "all",
        "answer_must_include": ["appliance", "zone", "peak", "drivers", "demand"],
        "answer_must_not_include": ["speculate", "I think", "probably"],
        "retrieval_notes": "Cross-scale peak driver analysis.",
        "evaluation_notes": "Answer must identify drivers from both micro (appliance) and macro (zone).",
    },
]
