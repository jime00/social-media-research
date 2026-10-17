import csv
import json
import math
from collections import Counter

input_path = 'data/global_student_digital_behavior_dataset.csv'
print(f"Auditing and cleaning {input_path}...")

total_rows = 0
dup_ids = 0
seen_ids = set()

null_counts = Counter()
countries = Counter()
genders = Counter()
devices = Counter()
income_levels = Counter()
urban_rural = Counter()
dev_levels = Counter()
education_levels = Counter()
fields_of_study = Counter()
brain_rot_levels = Counter()

# Quantitative trackers
numeric_cols = [
    'age', 'social_media_hours', 'study_hours_per_week', 'class_attendance_rate',
    'productivity_score', 'sleep_hours', 'brain_rot_index', 'digital_addiction_score',
    'academic_risk_score', 'financial_risk_score', 'attention_span_minutes',
    'stress_level', 'anxiety_score', 'depression_score', 'wellbeing_index',
    'sessions_per_day', 'average_session_length_minutes', 'late_night_score',
    'short_video_hours', 'entertainment_content_hours', 'education_content_hours',
    'news_content_hours', 'likes_given_per_day', 'comments_written_per_day',
    'posts_created_per_week', 'internet_access_hours', 'academic_motivation'
]

# We will collect sample values for exact percentiles/quantiles
sample_size = 50000
sample_numerics = {col: [] for col in numeric_cols}

sums = {col: 0.0 for col in numeric_cols}
mins = {col: float('inf') for col in numeric_cols}
maxs = {col: float('-inf') for col in numeric_cols}

zero_acad_risk = 0
pos_acad_risk = 0
pos_risk_vals = []

with open(input_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    
    for idx, row in enumerate(reader):
        total_rows += 1
        sid = row['student_id']
        if sid in seen_ids:
            dup_ids += 1
        seen_ids.add(sid)
        
        # Check missing values
        for k in fieldnames:
            v = row[k]
            if v == '' or v is None:
                null_counts[k] += 1
                
        # Categoricals
        countries[row['country']] += 1
        genders[row['gender']] += 1
        devices[row['device_access']] += 1
        income_levels[row['family_income_level']] += 1
        urban_rural[row['urban_rural']] += 1
        dev_levels[row['development_level']] += 1
        education_levels[row['education_level']] += 1
        fields_of_study[row['field_of_study']] += 1
        
        brl = row['brain_rot_level']
        if brl == '':
            brl = 'None'
        brain_rot_levels[brl] += 1
        
        ars = float(row['academic_risk_score'])
        if ars == 0.0:
            zero_acad_risk += 1
        else:
            pos_acad_risk += 1
            if len(pos_risk_vals) < 10000:
                pos_risk_vals.append(ars)
                
        # Numeric stats
        for col in numeric_cols:
            val = float(row[col])
            sums[col] += val
            if val < mins[col]: mins[col] = val
            if val > maxs[col]: maxs[col] = val
            if idx < sample_size:
                sample_numerics[col].append(val)

means = {col: sums[col] / total_rows for col in numeric_cols}

# Calculate quantiles from sample
def get_quantiles(arr):
    s = sorted(arr)
    n = len(s)
    return {
        'q1': s[n//4],
        'median': s[n//2],
        'q3': s[(3*n)//4],
        'iqr': s[(3*n)//4] - s[n//4]
    }

quantiles = {col: get_quantiles(sample_numerics[col]) for col in numeric_cols}

stats_summary = {
    'total_rows': total_rows,
    'duplicates': dup_ids,
    'null_counts': dict(null_counts),
    'academic_risk_score': {
        'zeros': zero_acad_risk,
        'zeros_pct': zero_acad_risk / total_rows * 100,
        'positives': pos_acad_risk,
        'positives_pct': pos_acad_risk / total_rows * 100,
        'pos_min': min(pos_risk_vals) if pos_risk_vals else 0,
        'pos_max': max(pos_risk_vals) if pos_risk_vals else 0,
        'pos_mean': sum(pos_risk_vals)/len(pos_risk_vals) if pos_risk_vals else 0
    },
    'categorical_distributions': {
        'genders': dict(genders),
        'devices': dict(devices),
        'development_levels': dict(dev_levels),
        'family_income_levels': dict(income_levels),
        'urban_rural': dict(urban_rural),
        'education_levels': dict(education_levels),
        'top_countries': dict(countries.most_common(10)),
        'total_countries': len(countries),
        'brain_rot_levels': dict(brain_rot_levels)
    },
    'numeric_metrics': {
        col: {
            'mean': round(means[col], 3),
            'min': round(mins[col], 3),
            'max': round(maxs[col], 3),
            'q1': round(quantiles[col]['q1'], 3),
            'median': round(quantiles[col]['median'], 3),
            'q3': round(quantiles[col]['q3'], 3),
            'iqr': round(quantiles[col]['iqr'], 3),
            'upper_bound': round(quantiles[col]['q3'] + 1.5 * quantiles[col]['iqr'], 3)
        }
        for col in numeric_cols
    }
}

with open('scratch/global_dataset_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats_summary, f, indent=2)

print("\n=== DATA CLEANING & AUDIT SUMMARY ===")
print(f"Total Rows: {total_rows}")
print(f"Duplicate IDs: {dup_ids}")
print(f"Nulls: {dict(null_counts)}")
print(f"academic_risk_score == 0: {zero_acad_risk} ({zero_acad_risk/total_rows*100:.2f}%)")
print(f"academic_risk_score > 0: {pos_acad_risk} ({pos_acad_risk/total_rows*100:.2f}%)")
print(f"Age: Mean={means['age']:.2f}, Range={mins['age']} to {maxs['age']}")
print(f"Social Media Hours: Mean={means['social_media_hours']:.2f}, Median={quantiles['social_media_hours']['median']:.2f}, Q1={quantiles['social_media_hours']['q1']:.2f}, Q3={quantiles['social_media_hours']['q3']:.2f}, IQR upper={quantiles['social_media_hours']['q3'] + 1.5*quantiles['social_media_hours']['iqr']:.2f}")
print(f"Study Hours / Week: Mean={means['study_hours_per_week']:.2f}, Median={quantiles['study_hours_per_week']['median']:.2f}")
print(f"Productivity Score: Mean={means['productivity_score']:.2f}, Median={quantiles['productivity_score']['median']:.2f}")
print(f"Class Attendance Rate: Mean={means['class_attendance_rate']:.2f}%, Median={quantiles['class_attendance_rate']['median']:.2f}%")
print(f"Sleep Hours: Mean={means['sleep_hours']:.2f}, Median={quantiles['sleep_hours']['median']:.2f}")
print(f"Brain Rot Index: Mean={means['brain_rot_index']:.2f}, Max={maxs['brain_rot_index']:.2f}")
print(f"Digital Addiction Score: Mean={means['digital_addiction_score']:.2f}, Max={maxs['digital_addiction_score']:.2f}")
print("Saved summary to scratch/global_dataset_stats.json")
