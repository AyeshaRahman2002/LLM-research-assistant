# Data Quality Report

Rows: **50**

## Columns

### participant_id
- dtype: `object`
- non-null: 50 / nulls: 0 (0.00%)
- top categories:
  - P001: 1
  - P012: 1
  - P024: 1
  - P003: 1
  - P004: 1
  - P005: 1
  - P006: 1
  - P007: 1
  - P008: 1
  - P009: 1

### age
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 22.0 / 35.700 / 55.0
- outliers (|z|>3): **0**

### gender
- dtype: `object`
- non-null: 49 / nulls: 1 (2.00%)
- top categories:
  - F: 24
  - M: 22
  - Other: 3

### education
- dtype: `object`
- non-null: 50 / nulls: 0 (0.00%)
- top categories:
  - Bachelor: 17
  - Master: 15
  - PhD: 13
  - High School: 5

### income_usd
- dtype: `float64`
- non-null: 47 / nulls: 3 (6.00%)
- min/mean/max: 31000.0 / 62212.766 / 95000.0
- outliers (|z|>3): **0**

### joined_date
- dtype: `object`
- non-null: 50 / nulls: 0 (0.00%)
- top categories:
  - 2023-05-10: 1
  - 2023-08-12: 1
  - 2022-08-14: 1
  - 2023-01-22: 1
  - 2021-08-09: 1
  - 2022-03-14: 1
  - 2023-07-01: 1
  - 2022-11-19: 1
  - 2020-05-05: 1
  - 2023-04-11: 1

### satisfaction_score
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 1.0 / 3.680 / 5.0
- outliers (|z|>3): **0**

### feedback_text
- dtype: `object`
- non-null: 50 / nulls: 0 (0.00%)
- top categories:
  - I like the explanations, but it's slow sometimes: 1
  - Limited context understanding: 1
  - It sometimes refuses simple tasks: 1
  - Responses are ok, not consistent: 1
  - Too many hallucinations: 1
  - Good for brainstorming: 1
  - Somewhat confusing interface: 1
  - Not relevant for my tasks: 1
  - Excellent! Especially summaries.: 1
  - Needs better reasoning.: 1

### uses_ai_tools
- dtype: `object`
- non-null: 50 / nulls: 0 (0.00%)
- top categories:
  - Yes: 17
  - Y: 12
  - True: 10
  - No: 7
  - False: 4

### preferred_model
- dtype: `object`
- non-null: 50 / nulls: 0 (0.00%)
- top categories:
  - gpt-4: 17
  - claude-3: 9
  - gpt-3.5: 7
  - vicuna-13b: 6
  - gemma-2b: 6
  - mistral-7b: 5

### completion_rate
- dtype: `float64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.39 / 0.797 / 1.0
- outliers (|z|>3): **0**

### satisfaction_score_score
- dtype: `float64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 1.0 / 3.680 / 5.0
- outliers (|z|>3): **0**

### gender__f
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.480 / 1.0
- outliers (|z|>3): **0**

### gender__m
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.440 / 1.0
- outliers (|z|>3): **0**

### gender__other
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.060 / 1.0
- outliers (|z|>3): **3**

### education__bachelor
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.340 / 1.0
- outliers (|z|>3): **0**

### education__high school
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.100 / 1.0
- outliers (|z|>3): **0**

### education__master
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.300 / 1.0
- outliers (|z|>3): **0**

### education__phd
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.260 / 1.0
- outliers (|z|>3): **0**

### preferred_model__claude-3
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.180 / 1.0
- outliers (|z|>3): **0**

### preferred_model__gemma-2b
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.120 / 1.0
- outliers (|z|>3): **0**

### preferred_model__gpt-3.5
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.140 / 1.0
- outliers (|z|>3): **0**

### preferred_model__gpt-4
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.340 / 1.0
- outliers (|z|>3): **0**

### preferred_model__mistral-7b
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.100 / 1.0
- outliers (|z|>3): **0**

### preferred_model__vicuna-13b
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.120 / 1.0
- outliers (|z|>3): **0**

### uses_ai_tools__false
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.080 / 1.0
- outliers (|z|>3): **4**

### uses_ai_tools__no
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.140 / 1.0
- outliers (|z|>3): **0**

### uses_ai_tools__true
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.200 / 1.0
- outliers (|z|>3): **0**

### uses_ai_tools__y
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.240 / 1.0
- outliers (|z|>3): **0**

### uses_ai_tools__yes
- dtype: `int64`
- non-null: 50 / nulls: 0 (0.00%)
- min/mean/max: 0.0 / 0.340 / 1.0
- outliers (|z|>3): **0**
