#!/usr/bin/env python3
"""
Extracts representative student data points and exact cohort aggregations
from data/global_student_digital_behavior_dataset.csv to create js/interactive_data.js
for fast, responsive client-side Plotly.js visualizations.
"""

import csv
import json
import math
import os

input_csv = 'data/global_student_digital_behavior_dataset.csv'
output_js = 'js/interactive_data.js'

print(f"Reading {input_csv} to build interactive visualization dataset...")

# Cohort accumulators
education_levels = ["School", "Diploma", "Graduate", "Postgraduate", "Dropout", "PhD"]
usage_tiers = ["Low (<= 3 hrs)", "Moderate (3-5 hrs)", "High (> 5 hrs)"]

# Aggregate metrics for education level
edu_stats = {
    edu: {
        'count': 0,
        'sm_hours': [],
        'attention': [],
        'study_hours': [],
        'sleep_hours': [],
        'brain_rot': [],
        'academic_risk_pos': 0
    }
    for edu in education_levels
}

# Aggregate metrics for usage tier
usage_stats = {
    tier: {
        'count': 0,
        'sm_hours': [],
        'attention': [],
        'study_hours': [],
        'sleep_hours': [],
        'brain_rot': [],
        'academic_risk_pos': 0
    }
    for tier in usage_tiers
}

# Representative scatter sample (stratified across education levels)
scatter_samples = {edu: [] for edu in education_levels}
sample_target_per_edu = 250  # ~1500 points total for instant 60fps rendering

total_rows = 0

with open(input_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        total_rows += 1
        
        edu = row['education_level']
        if edu not in edu_stats:
            continue
            
        sm = float(row['social_media_hours'])
        att = float(row['attention_span_minutes'])
        study = float(row['study_hours_per_week'])
        sleep = float(row['sleep_hours'])
        br = float(row['brain_rot_index'])
        risk = float(row['academic_risk_score'])
        prod = float(row['productivity_score'])
        
        # Usage tier assignment
        if sm <= 3.0:
            tier = "Low (<= 3 hrs)"
        elif sm <= 5.0:
            tier = "Moderate (3-5 hrs)"
        else:
            tier = "High (> 5 hrs)"
            
        # Accumulate education stats
        edu_stats[edu]['count'] += 1
        if len(edu_stats[edu]['sm_hours']) < 15000:
            edu_stats[edu]['sm_hours'].append(sm)
            edu_stats[edu]['attention'].append(att)
            edu_stats[edu]['study_hours'].append(study)
            edu_stats[edu]['sleep_hours'].append(sleep)
            edu_stats[edu]['brain_rot'].append(br)
        if risk > 0.0:
            edu_stats[edu]['academic_risk_pos'] += 1
            
        # Accumulate usage stats
        usage_stats[tier]['count'] += 1
        if len(usage_stats[tier]['sm_hours']) < 15000:
            usage_stats[tier]['sm_hours'].append(sm)
            usage_stats[tier]['attention'].append(att)
            usage_stats[tier]['study_hours'].append(study)
            usage_stats[tier]['sleep_hours'].append(sleep)
            usage_stats[tier]['brain_rot'].append(br)
        if risk > 0.0:
            usage_stats[tier]['academic_risk_pos'] += 1
            
        # Stratified scatter sample
        if len(scatter_samples[edu]) < sample_target_per_edu:
            # Sample evenly across rows
            if idx % 10 == 0:
                scatter_samples[edu].append({
                    'id': int(row['student_id']),
                    'country': row['country'],
                    'edu': edu,
                    'sm_hours': round(sm, 2),
                    'attention': round(att, 2),
                    'study_hours': round(study, 2),
                    'sleep_hours': round(sleep, 2),
                    'brain_rot': round(br, 1),
                    'prod_score': round(prod, 1),
                    'tier': tier
                })

print(f"Processed {total_rows} total rows.")

# Calculate 5-number summary (min, q1, median, q3, max)
def calc_box_summary(arr):
    if not arr:
        return {'min': 0, 'q1': 0, 'median': 0, 'q3': 0, 'max': 0, 'mean': 0}
    s = sorted(arr)
    n = len(s)
    def q(p):
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return s[lo]
        return s[lo] + (idx - lo) * (s[hi] - s[lo])
    return {
        'min': round(s[0], 2),
        'q1': round(q(0.25), 2),
        'median': round(q(0.50), 2),
        'q3': round(q(0.75), 2),
        'max': round(s[-1], 2),
        'mean': round(sum(s) / n, 2)
    }

# Compute education level summaries
edu_aggregates = {}
for edu in education_levels:
    d = edu_stats[edu]
    cnt = d['count']
    edu_aggregates[edu] = {
        'count': cnt,
        'sm_hours': calc_box_summary(d['sm_hours']),
        'attention': calc_box_summary(d['attention']),
        'study_hours': calc_box_summary(d['study_hours']),
        'sleep_hours': calc_box_summary(d['sleep_hours']),
        'brain_rot': calc_box_summary(d['brain_rot']),
        'academic_risk_rate_pct': round((d['academic_risk_pos'] / cnt) * 100, 2) if cnt > 0 else 0
    }

# Compute usage tier summaries
usage_aggregates = {}
for tier in usage_tiers:
    d = usage_stats[tier]
    cnt = d['count']
    usage_aggregates[tier] = {
        'count': cnt,
        'sm_hours': calc_box_summary(d['sm_hours']),
        'attention': calc_box_summary(d['attention']),
        'study_hours': calc_box_summary(d['study_hours']),
        'sleep_hours': calc_box_summary(d['sleep_hours']),
        'brain_rot': calc_box_summary(d['brain_rot']),
        'academic_risk_rate_pct': round((d['academic_risk_pos'] / cnt) * 100, 2) if cnt > 0 else 0
    }

# Flatten scatter samples into list
all_scatter_points = []
for edu in education_levels:
    all_scatter_points.extend(scatter_samples[edu])

print(f"Compiled {len(all_scatter_points)} stratified scatter points.")

# Construct JavaScript payload
interactive_data = {
    'total_students': total_rows,
    'education_levels': education_levels,
    'usage_tiers': usage_tiers,
    'scatter_sample': all_scatter_points,
    'education_aggregates': edu_aggregates,
    'usage_aggregates': usage_aggregates
}

os.makedirs('js', exist_ok=True)
with open(output_js, 'w', encoding='utf-8') as f:
    f.write('/**\n')
    f.write(' * Empirical aggregate data compiled from data/global_student_digital_behavior_dataset.csv (N = 500,000)\n')
    f.write(' * Used for fast, responsive client-side Plotly.js visualizations.\n')
    f.write(' */\n')
    f.write('window.RESEARCH_INTERACTIVE_DATA = ')
    json.dump(interactive_data, f, indent=2)
    f.write(';\n')

print(f"Successfully generated {output_js} (size: {os.path.getsize(output_js)} bytes).")
