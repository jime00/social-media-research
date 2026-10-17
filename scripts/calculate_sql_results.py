import csv
import json

path = 'data/global_student_digital_behavior_dataset.csv'

# Country mapping
countries = {}
# Query metrics
q1_samples = []
q2_count = 0
q3_count = 0
q4_top5 = []
q5_stats = {
    'total': 0,
    'avg_sm': 0.0,
    'avg_study': 0.0,
    'min_sm': 999.0,
    'max_sm': -999.0,
    'avg_prod': 0.0
}
q6_groups = {'Low': {'count': 0, 'study': 0.0, 'prod': 0.0, 'addict': 0.0},
             'Moderate': {'count': 0, 'study': 0.0, 'prod': 0.0, 'addict': 0.0},
             'High': {'count': 0, 'study': 0.0, 'prod': 0.0, 'addict': 0.0}}

country_stats = {}
dev_stats = {'Developing': {'count': 0, 'sm': 0.0, 'study': 0.0, 'prod': 0.0},
             'Developed': {'count': 0, 'sm': 0.0, 'study': 0.0, 'prod': 0.0},
             'Underdeveloped': {'count': 0, 'sm': 0.0, 'study': 0.0, 'prod': 0.0}}

with open(path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, r in enumerate(reader):
        sid = int(r['student_id'])
        cname = r['country']
        dev = r['development_level']
        age = int(float(r['age']))
        gender = r['gender']
        sm = float(r['social_media_hours'])
        study = float(r['study_hours_per_week'])
        prod = float(r['productivity_score'])
        addict = float(r['digital_addiction_score'])
        mot = float(r['academic_motivation'])
        
        tier = 'Low' if sm <= 3.0 else ('Moderate' if sm <= 5.0 else 'High')
        
        # Q1: First 5
        if i < 5:
            q1_samples.append({
                'student_id': sid, 'age': age, 'gender': gender,
                'social_media_hours': sm, 'study_hours_per_week': study, 'productivity_score': prod
            })
            
        # Q2: WHERE sm > 6.0
        if sm > 6.0:
            q2_count += 1
            
        # Q3: WHERE sm > 5.0 AND addict > 20.0 AND mot < 4.0
        if sm > 5.0 and addict > 20.0 and mot < 4.0:
            q3_count += 1
            
        # Q5 totals
        q5_stats['total'] += 1
        q5_stats['avg_sm'] += sm
        q5_stats['avg_study'] += study
        q5_stats['avg_prod'] += prod
        if sm < q5_stats['min_sm']: q5_stats['min_sm'] = sm
        if sm > q5_stats['max_sm']: q5_stats['max_sm'] = sm
        
        # Q6
        q6_groups[tier]['count'] += 1
        q6_groups[tier]['study'] += study
        q6_groups[tier]['prod'] += prod
        q6_groups[tier]['addict'] += addict
        
        # Country aggregation
        if cname not in country_stats:
            country_stats[cname] = {'count': 0, 'sm': 0.0, 'study': 0.0}
        country_stats[cname]['count'] += 1
        country_stats[cname]['sm'] += sm
        country_stats[cname]['study'] += study
        
        # Join aggregation by dev level
        if dev in dev_stats:
            dev_stats[dev]['count'] += 1
            dev_stats[dev]['sm'] += sm
            dev_stats[dev]['study'] += study
            dev_stats[dev]['prod'] += prod

# Q4: Top 5 study hours (sorted from top rows)
# Run a quick top 5
print(f"Total processed: {q5_stats['total']}")
print(f"Q2 (Screen time > 6.0 hrs): {q2_count} students ({(q2_count/500000)*100:.2f}%)")
print(f"Q3 (Multiple conditions: sm>5, addict>20, mot<4): {q3_count} students ({(q3_count/500000)*100:.2f}%)")
print(f"Q5: Avg SM: {q5_stats['avg_sm']/500000:.2f}h, Avg Study: {q5_stats['avg_study']/500000:.2f}h, Avg Prod: {q5_stats['avg_prod']/500000:.2f}")

for t in ['Low', 'Moderate', 'High']:
    c = q6_groups[t]['count']
    print(f"Q6 Tier {t}: Count={c} ({(c/500000)*100:.1f}%), Avg Study={q6_groups[t]['study']/c:.2f}h, Avg Prod={q6_groups[t]['prod']/c:.2f}, Avg Addict={q6_groups[t]['addict']/c:.2f}")

# Q7 HAVING avg_sm > 3.40
q7_results = []
for cname, s in country_stats.items():
    avg_s = s['sm'] / s['count']
    if avg_s > 3.40:
        q7_results.append((cname, s['count'], avg_s, s['study'] / s['count']))
q7_results.sort(key=lambda x: x[2], reverse=True)
print(f"Q7 Countries with Avg SM > 3.40h (Count: {len(q7_results)}):")
for r in q7_results[:6]:
    print(f"  {r[0]}: N={r[1]}, Avg SM={r[2]:.2f}h, Avg Study={r[3]:.2f}h")

# Q8 JOIN dev levels
print("Q8 Development Level JOIN results:")
for dev, s in dev_stats.items():
    c = s['count']
    print(f"  {dev}: N={c} ({(c/500000)*100:.1f}%), Avg SM={s['sm']/c:.2f}h, Avg Study={s['study']/c:.2f}h, Avg Prod={s['prod']/c:.2f}")

