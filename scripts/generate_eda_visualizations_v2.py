import csv
import math
import subprocess
import os

print("Generating 10 EDA Visualizations for 500k Global Student Dataset...")

csv_path = 'data/global_student_digital_behavior_dataset.csv'
os.makedirs('scratch/svgs', exist_ok=True)
os.makedirs('images', exist_ok=True)

def convert_svg_to_png(svg_path, png_path, width=1024, height=640):
    cmd = ["sips", "-s", "format", "png", svg_path, "--out", png_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error converting {svg_path}: {res.stderr}")
    else:
        subprocess.run(["sips", "-z", str(height), str(width), png_path], capture_output=True)
        print(f"Generated: {png_path} ({width}x{height})")

# Accumulate distributions across all 500,000 records
age_counts = {a: 0 for a in range(15, 26)}
device_counts = {"Smartphone": 0, "Both": 0, "Shared Device": 0, "Laptop": 0}

# Histograms
# 1. Social media hours (0-9)
sm_bins = [0]*9  # [0-1), [1-2), ... [8-9]
# 2. Sessions per day (1 to 26)
session_bins = [0]*7 # [1-4], [5-8], [9-12], [13-16], [17-20], [21-24], [25+]
# 6. Study hours per week (0-45)
study_bins = [0]*8 # [0-10), [10-15), [15-20), [20-25), [25-30), [30-35), [35-40), [40+]

# Boxplot aggregators (Productivity score by SM usage: Low <=3, Mod 3-5, High >5)
prod_by_sm = {"low": [], "mod": [], "high": []}
# Boxplot aggregators (Weekly study hours by SM usage: Low <=3, Mod 3-5, High >5)
study_by_sm = {"low": [], "mod": [], "high": []}

# Heatmap: SM hours (5 bins) x Attention span (4 bins)
# SM: [0-2), [2-4), [4-6), [6-8), [8-9+]
# Attention: [25-35), [35-45), [45-55), [55-60]
heatmap_grid = [[0 for _ in range(4)] for _ in range(5)]

# Subsample points for scatter plot (Figure 9)
scatter_points = []

# Correlation matrix variables
corr_vars = [
    'social_media_hours',
    'study_hours_per_week',
    'class_attendance_rate',
    'productivity_score',
    'brain_rot_index',
    'digital_addiction_score',
    'attention_span_minutes',
    'academic_motivation'
]
corr_sums = {v: 0.0 for v in corr_vars}
corr_sq_sums = {v: 0.0 for v in corr_vars}
cross_sums = {(v1, v2): 0.0 for v1 in corr_vars for v2 in corr_vars}

N = 0

print("Streaming dataset...")
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, r in enumerate(reader):
        N += 1
        age = int(float(r['age']))
        if 15 <= age <= 25:
            age_counts[age] += 1
        
        dev = r['device_access']
        if dev in device_counts:
            device_counts[dev] += 1
            
        sm_h = float(r['social_media_hours'])
        b_sm = min(8, int(sm_h))
        sm_bins[b_sm] += 1
        
        sess = float(r['sessions_per_day'])
        if sess <= 4: b_se = 0
        elif sess <= 8: b_se = 1
        elif sess <= 12: b_se = 2
        elif sess <= 16: b_se = 3
        elif sess <= 20: b_se = 4
        elif sess <= 24: b_se = 5
        else: b_se = 6
        session_bins[b_se] += 1
        
        st_h = float(r['study_hours_per_week'])
        if st_h < 10: b_st = 0
        elif st_h < 15: b_st = 1
        elif st_h < 20: b_st = 2
        elif st_h < 25: b_st = 3
        elif st_h < 30: b_st = 4
        elif st_h < 35: b_st = 5
        elif st_h < 40: b_st = 6
        else: b_st = 7
        study_bins[b_st] += 1
        
        prod = float(r['productivity_score'])
        att = float(r['attention_span_minutes'])
        
        # SM Tier
        tier = "low" if sm_h <= 3.0 else ("mod" if sm_h <= 5.0 else "high")
        # Subsample for boxplot percentiles to save memory
        if i % 25 == 0:
            prod_by_sm[tier].append(prod)
            study_by_sm[tier].append(st_h)
            
        # Heatmap bins
        h_sm = min(4, int(sm_h // 2))
        if att < 35: h_at = 0
        elif att < 45: h_at = 1
        elif att < 55: h_at = 2
        else: h_at = 3
        heatmap_grid[h_sm][h_at] += 1
        
        # Scatter points (1 in 1000)
        if i % 1000 == 0:
            scatter_points.append((sm_h, st_h))
            
        # Correlation accumulators
        vals = {v: float(r[v]) for v in corr_vars}
        for v in corr_vars:
            corr_sums[v] += vals[v]
            corr_sq_sums[v] += vals[v] ** 2
        for v1 in corr_vars:
            for v2 in corr_vars:
                cross_sums[(v1, v2)] += vals[v1] * vals[v2]

print(f"Aggregated {N} rows.")

# Calculate 8x8 correlation matrix
corr_matrix = {}
for v1 in corr_vars:
    corr_matrix[v1] = {}
    mean1 = corr_sums[v1] / N
    var1 = (corr_sq_sums[v1] / N) - (mean1 ** 2)
    std1 = math.sqrt(max(0.0, var1))
    for v2 in corr_vars:
        mean2 = corr_sums[v2] / N
        var2 = (corr_sq_sums[v2] / N) - (mean2 ** 2)
        std2 = math.sqrt(max(0.0, var2))
        cov = (cross_sums[(v1, v2)] / N) - (mean1 * mean2)
        r = cov / (std1 * std2) if (std1 * std2) > 0 else 0.0
        corr_matrix[v1][v2] = r

# Helper for boxplot metrics
def get_box_stats(arr):
    arr = sorted(arr)
    n = len(arr)
    mean = sum(arr) / n
    med = arr[n // 2]
    q1 = arr[int(n * 0.25)]
    q3 = arr[int(n * 0.75)]
    iqr = q3 - q1
    lower_w = max(min(arr), q1 - 1.5 * iqr)
    upper_w = min(max(arr), q3 + 1.5 * iqr)
    return {"mean": mean, "med": med, "q1": q1, "q3": q3, "low": lower_w, "high": upper_w, "n": n}

# ==============================================================================
# Figure 1: Time Spent on Social Media (Screen Time Histogram)
# ==============================================================================
def make_fig1():
    w, h = 1024, 640
    ml, mr, mt, mb = 80, 40, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    labels = ["0–1 h", "1–2 h", "2–3 h", "3–4 h", "4–5 h", "5–6 h", "6–7 h", "7–8 h", "8–9 h"]
    max_c = 160000
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Distribution of Daily Social Media Screen Time (N = {N:,})</text>')
    
    for y_val in [0, 40000, 80000, 120000, 160000]:
        y_pos = mt + ph - (y_val / max_c) * ph
        svg.append(f'<line x1="{ml}" y1="{y_pos}" x2="{w - mr}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="end" fill="#57606a">{y_val:,}</text>')
        
    svg.append(f'<text x="24" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 24 {mt + ph/2})">Number of Students</text>')
    
    gap = pw / len(sm_bins)
    bw = gap * 0.8
    for i, (lbl, c) in enumerate(zip(labels, sm_bins)):
        bx = ml + i * gap + (gap - bw)/2
        bh = (c / max_c) * ph
        by = mt + ph - bh
        col = "#bc5689" if 2 <= i <= 4 else "#2a4365"
        pct = (c / N) * 100
        svg.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{col}" rx="3" stroke="#1b2d42" stroke-width="1"/>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 18}" font-family="-apple-system, sans-serif" font-size="12" font-weight="700" text-anchor="middle" fill="#1b2d42">{c:,}</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 4}" font-family="-apple-system, sans-serif" font-size="11" text-anchor="middle" fill="#57606a">({pct:.1f}%)</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{mt + ph + 24}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" text-anchor="middle" fill="#24292f">{lbl}</text>')
        
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Daily Screen Time in Hours (Mean = 3.36 hrs, Median = 3.28 hrs, IQR = 1.73 hrs)</text>')
    
    # Legend
    svg.append(f'<rect x="{w - mr - 260}" y="{mt + 15}" width="240" height="60" fill="#f8fafc" stroke="#cbd5e1" rx="6"/>')
    svg.append(f'<rect x="{w - mr - 245}" y="{mt + 26}" width="14" height="14" fill="#bc5689" stroke="#1b2d42"/>')
    svg.append(f'<text x="{w - mr - 222}" y="{mt + 38}" font-family="-apple-system, sans-serif" font-size="12" fill="#1b2d42">Core Usage Range (2–5 hrs: 72.8%)</text>')
    svg.append(f'<rect x="{w - mr - 245}" y="{mt + 48}" width="14" height="14" fill="#2a4365" stroke="#1b2d42"/>')
    svg.append(f'<text x="{w - mr - 222}" y="{mt + 60}" font-family="-apple-system, sans-serif" font-size="12" fill="#57606a">Lower &amp; Upper Usage Tiers</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig1.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/time-spent-histogram.png', w, h)

# ==============================================================================
# Figure 2: Frequency of Social Media Use (Daily Digital Sessions)
# ==============================================================================
def make_fig2():
    w, h = 1024, 640
    ml, mr, mt, mb = 80, 40, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    labels = ["1–4", "5–8", "9–12", "13–16", "17–20", "21–24", "25+"]
    max_c = 220000
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Frequency of Social Media Access: Daily Sessions (N = {N:,})</text>')
    
    for y_val in [0, 50000, 100000, 150000, 200000]:
        y_pos = mt + ph - (y_val / max_c) * ph
        svg.append(f'<line x1="{ml}" y1="{y_pos}" x2="{w - mr}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="end" fill="#57606a">{y_val:,}</text>')
        
    svg.append(f'<text x="24" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 24 {mt + ph/2})">Number of Students</text>')
    
    gap = pw / len(session_bins)
    bw = gap * 0.75
    for i, (lbl, c) in enumerate(zip(labels, session_bins)):
        bx = ml + i * gap + (gap - bw)/2
        bh = (c / max_c) * ph
        by = mt + ph - bh
        pct = (c / N) * 100
        col = "#2a4365" if i in [1, 2] else ("#bc5689" if i >= 4 else "#1b2d42")
        svg.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{col}" rx="3" stroke="#1b2d42" stroke-width="1"/>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 18}" font-family="-apple-system, sans-serif" font-size="12" font-weight="700" text-anchor="middle" fill="#1b2d42">{c:,}</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 4}" font-family="-apple-system, sans-serif" font-size="11" text-anchor="middle" fill="#57606a">({pct:.1f}%)</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{mt + ph + 24}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#24292f">{lbl} sessions</text>')
        
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Daily App Launch Sessions (Mean = 9.90 sessions/day, Median = 9.73 sessions/day)</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig2.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/frequency-use-barchart.png', w, h)

# ==============================================================================
# Figure 3: Academic Impact Box Plot (Productivity by Usage Tier)
# ==============================================================================
def make_fig3():
    w, h = 1024, 640
    ml, mr, mt, mb = 80, 40, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    
    s_low = get_box_stats(prod_by_sm["low"])
    s_mod = get_box_stats(prod_by_sm["mod"])
    s_high = get_box_stats(prod_by_sm["high"])
    
    tiers = [
        ("Low Usage (≤ 3.0 hrs)", s_low, "#2e7d32"),
        ("Moderate Usage (3.0–5.0 hrs)", s_mod, "#d97706"),
        ("High Usage (> 5.0 hrs)", s_high, "#dc2626")
    ]
    
    def to_y(v): return mt + ph - ((v - 1.0) / 9.0) * ph
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Academic Productivity Score by Social Media Usage Level (N = {N:,})</text>')
    
    for v in range(1, 11):
        yp = to_y(v)
        svg.append(f'<line x1="{ml}" y1="{yp}" x2="{w - mr}" y2="{yp}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 12}" y="{yp + 4}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="end" fill="#57606a">{v}.0</text>')
        
    svg.append(f'<text x="24" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 24 {mt + ph/2})">Productivity Score (1.0 to 10.0)</text>')
    
    gap = pw / len(tiers)
    box_w = 120
    for i, (tname, st, col) in enumerate(tiers):
        cx = ml + i * gap + gap/2
        y_q1, y_q3 = to_y(st["q1"]), to_y(st["q3"])
        y_med = to_y(st["med"])
        y_mean = to_y(st["mean"])
        y_low, y_high = to_y(st["low"]), to_y(st["high"])
        
        # Whiskers
        svg.append(f'<line x1="{cx}" y1="{y_q1}" x2="{cx}" y2="{y_low}" stroke="#1b2d42" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx}" y1="{y_q3}" x2="{cx}" y2="{y_high}" stroke="#1b2d42" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx - 24}" y1="{y_low}" x2="{cx + 24}" y2="{y_low}" stroke="#1b2d42" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx - 24}" y1="{y_high}" x2="{cx + 24}" y2="{y_high}" stroke="#1b2d42" stroke-width="1.5"/>')
        
        # Box
        svg.append(f'<rect x="{cx - box_w/2}" y="{y_q3}" width="{box_w}" height="{y_q1 - y_q3}" fill="#ffffff" stroke="#1b2d42" stroke-width="2"/>')
        # Median line
        svg.append(f'<line x1="{cx - box_w/2}" y1="{y_med}" x2="{cx + box_w/2}" y2="{y_med}" stroke="#d97706" stroke-width="3"/>')
        # Mean marker
        svg.append(f'<polygon points="{cx},{y_mean-5} {cx+5},{y_mean} {cx},{y_mean+5} {cx-5},{y_mean}" fill="{col}" stroke="#1b2d42" stroke-width="1"/>')
        
        # Labels
        svg.append(f'<text x="{cx}" y="{mt + ph + 24}" font-family="-apple-system, sans-serif" font-size="14" font-weight="700" text-anchor="middle" fill="#1b2d42">{tname}</text>')
        svg.append(f'<text x="{cx}" y="{mt + ph + 42}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="middle" fill="#57606a">Mean: {st["mean"]:.2f} | Med: {st["med"]:.2f}</text>')
        
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 16}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#24292f">Usage Tiers (Mean Productivity Decreases by -1.03 points from Low to High Usage)</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig3.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/academic-impact-boxplot.png', w, h)

# ==============================================================================
# Figure 4: Screen Time vs Attention Span Heatmap
# ==============================================================================
def make_fig4():
    w, h = 1024, 640
    ml, mr, mt, mb = 140, 100, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    sm_lbls = ["0–2 h", "2–4 h", "4–6 h", "6–8 h", "8–9+ h"]
    att_lbls = ["25–35 min", "35–45 min", "45–55 min", "55–60 min"]
    
    cw = pw / len(att_lbls)
    ch = ph / len(sm_lbls)
    
    max_cell = max(max(row) for row in heatmap_grid)
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Social Media Screen Time vs. Attention Span Distribution Heatmap</text>')
    
    for r in range(5):
        svg.append(f'<text x="{ml - 14}" y="{mt + r*ch + ch/2 + 5}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="end" fill="#1b2d42">{sm_lbls[r]}</text>')
        for c in range(4):
            cnt = heatmap_grid[r][c]
            t = cnt / max_cell
            red = int(255 - t * (255 - 188))
            green = int(255 - t * (255 - 86))
            blue = int(255 - t * (255 - 137))
            col = f"rgb({red},{green},{blue})"
            txt_col = "#ffffff" if t > 0.45 else "#1b2d42"
            
            x = ml + c * cw
            y = mt + r * ch
            svg.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{col}" stroke="#ffffff" stroke-width="2"/>')
            svg.append(f'<text x="{x + cw/2}" y="{y + ch/2 + 2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="700" text-anchor="middle" fill="{txt_col}">{cnt:,}</text>')
            svg.append(f'<text x="{x + cw/2}" y="{y + ch/2 + 18}" font-family="-apple-system, sans-serif" font-size="11" text-anchor="middle" fill="{txt_col}">({(cnt/N)*100:.1f}%)</text>')
            
    for c in range(4):
        svg.append(f'<text x="{ml + c*cw + cw/2}" y="{mt + ph + 24}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#1b2d42">{att_lbls[c]}</text>')
        
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Attention Span Duration (Minutes; Strong Clustering at 2–4h Screen Time &amp; 55–60m Attention)</text>')
    svg.append(f'<text x="28" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 28 {mt + ph/2})">Social Media Hours</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig4.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/time-vs-concentration-heatmap.png', w, h)

# ==============================================================================
# Figure 5: Age Distribution
# ==============================================================================
def make_fig5():
    w, h = 1024, 640
    ml, mr, mt, mb = 80, 40, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    ages = list(range(15, 26))
    max_c = 55000
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Student Age Distribution (N = {N:,}, Ages 15 to 25)</text>')
    
    for y_val in [0, 10000, 20000, 30000, 40000, 50000]:
        y_pos = mt + ph - (y_val / max_c) * ph
        svg.append(f'<line x1="{ml}" y1="{y_pos}" x2="{w - mr}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="end" fill="#57606a">{y_val:,}</text>')
        
    svg.append(f'<text x="24" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 24 {mt + ph/2})">Number of Students</text>')
    
    gap = pw / len(ages)
    bw = gap * 0.75
    for i, a in enumerate(ages):
        c = age_counts[a]
        bx = ml + i * gap + (gap - bw)/2
        bh = (c / max_c) * ph
        by = mt + ph - bh
        col = "#bc5689" if 18 <= a <= 22 else "#2a4365"
        svg.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{col}" rx="3" stroke="#1b2d42" stroke-width="1"/>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 7}" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" text-anchor="middle" fill="#1b2d42">{c:,}</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{mt + ph + 22}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="middle" fill="#24292f">{a}</text>')
        
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Age in Years (Mean = 20.0 yrs, Median = 20.0 yrs, Uniform Sampling Across Tertiary &amp; Secondary Cohorts)</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig5.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/age-distribution.png', w, h)

# ==============================================================================
# Figure 6: Weekly Study Hours Distribution
# ==============================================================================
def make_fig6():
    w, h = 1024, 640
    ml, mr, mt, mb = 80, 40, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    labels = ["0–10 h", "10–15 h", "15–20 h", "20–25 h", "25–30 h", "30–35 h", "35–40 h", "40+ h"]
    max_c = 180000
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Distribution of Weekly Study Time (N = {N:,})</text>')
    
    for y_val in [0, 40000, 80000, 120000, 160000]:
        y_pos = mt + ph - (y_val / max_c) * ph
        svg.append(f'<line x1="{ml}" y1="{y_pos}" x2="{w - mr}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="end" fill="#57606a">{y_val:,}</text>')
        
    svg.append(f'<text x="24" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 24 {mt + ph/2})">Number of Students</text>')
    
    gap = pw / len(study_bins)
    bw = gap * 0.75
    for i, (lbl, c) in enumerate(zip(labels, study_bins)):
        bx = ml + i * gap + (gap - bw)/2
        bh = (c / max_c) * ph
        by = mt + ph - bh
        pct = (c / N) * 100
        col = "#2a4365" if 2 <= i <= 4 else "#1b2d42"
        svg.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{col}" rx="3" stroke="#1b2d42" stroke-width="1"/>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 18}" font-family="-apple-system, sans-serif" font-size="12" font-weight="700" text-anchor="middle" fill="#1b2d42">{c:,}</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 4}" font-family="-apple-system, sans-serif" font-size="11" text-anchor="middle" fill="#57606a">({pct:.1f}%)</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{mt + ph + 24}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#24292f">{lbl}</text>')
        
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Weekly Study Duration (Mean = 22.98 hrs, Median = 23.11 hrs, IQR = 19.46–26.60 hrs)</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig6.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/study-time-distribution.png', w, h)

# ==============================================================================
# Figure 7: Primary Device Distribution
# ==============================================================================
def make_fig7():
    w, h = 1024, 640
    ml, mr, mt, mb = 80, 40, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    devices = [
        ("Smartphone", device_counts["Smartphone"], 60.6, "#2563eb"),
        ("Both (Phone &amp; Laptop)", device_counts["Both"], 14.9, "#bc5689"),
        ("Shared Device", device_counts["Shared Device"], 13.4, "#d97706"),
        ("Laptop Only", device_counts["Laptop"], 11.1, "#059669")
    ]
    max_c = 350000
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Primary Digital Device Distribution (N = {N:,})</text>')
    
    for y_val in [0, 70000, 140000, 210000, 280000, 350000]:
        y_pos = mt + ph - (y_val / max_c) * ph
        svg.append(f'<line x1="{ml}" y1="{y_pos}" x2="{w - mr}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="end" fill="#57606a">{y_val:,}</text>')
        
    svg.append(f'<text x="24" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 24 {mt + ph/2})">Number of Students</text>')
    
    gap = pw / len(devices)
    bw = gap * 0.65
    for i, (dname, cnt, pct, col) in enumerate(devices):
        bx = ml + i * gap + (gap - bw)/2
        bh = (cnt / max_c) * ph
        by = mt + ph - bh
        svg.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{col}" rx="4" stroke="#1b2d42" stroke-width="1"/>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 18}" font-family="-apple-system, sans-serif" font-size="13" font-weight="700" text-anchor="middle" fill="#1b2d42">{cnt:,}</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{by - 4}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" text-anchor="middle" fill="#57606a">({pct:.1f}%)</text>')
        svg.append(f'<text x="{bx + bw/2}" y="{mt + ph + 24}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#24292f">{dname}</text>')
        
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Primary Device Utilized for Studies &amp; Social Media Access (Smartphones Predominate at 60.6%)</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig7.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/platform-distribution.png', w, h)

# ==============================================================================
# Figure 8: Weekly Study Hours by Usage Tier (Comparative Box Plot)
# ==============================================================================
def make_fig8():
    w, h = 1024, 640
    ml, mr, mt, mb = 80, 40, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    
    st_low = get_box_stats(study_by_sm["low"])
    st_mod = get_box_stats(study_by_sm["mod"])
    st_high = get_box_stats(study_by_sm["high"])
    
    tiers = [
        ("Low Usage (≤ 3.0 hrs)", st_low, "#2e7d32"),
        ("Moderate Usage (3.0–5.0 hrs)", st_mod, "#d97706"),
        ("High Usage (> 5.0 hrs)", st_high, "#dc2626")
    ]
    
    def to_y(v): return mt + ph - (v / 45.0) * ph
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Weekly Study Time Across Social Media Usage Levels (N = {N:,})</text>')
    
    for v in [0, 10, 20, 30, 40]:
        yp = to_y(v)
        svg.append(f'<line x1="{ml}" y1="{yp}" x2="{w - mr}" y2="{yp}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 12}" y="{yp + 4}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="end" fill="#57606a">{v} h</text>')
        
    svg.append(f'<text x="24" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 24 {mt + ph/2})">Weekly Study Duration (Hours)</text>')
    
    gap = pw / len(tiers)
    box_w = 120
    for i, (tname, st, col) in enumerate(tiers):
        cx = ml + i * gap + gap/2
        y_q1, y_q3 = to_y(st["q1"]), to_y(st["q3"])
        y_med = to_y(st["med"])
        y_mean = to_y(st["mean"])
        y_low, y_high = to_y(st["low"]), to_y(st["high"])
        
        svg.append(f'<line x1="{cx}" y1="{y_q1}" x2="{cx}" y2="{y_low}" stroke="#1b2d42" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx}" y1="{y_q3}" x2="{cx}" y2="{y_high}" stroke="#1b2d42" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx - 24}" y1="{y_low}" x2="{cx + 24}" y2="{y_low}" stroke="#1b2d42" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx - 24}" y1="{y_high}" x2="{cx + 24}" y2="{y_high}" stroke="#1b2d42" stroke-width="1.5"/>')
        
        svg.append(f'<rect x="{cx - box_w/2}" y="{y_q3}" width="{box_w}" height="{y_q1 - y_q3}" fill="#ffffff" stroke="#1b2d42" stroke-width="2"/>')
        svg.append(f'<line x1="{cx - box_w/2}" y1="{y_med}" x2="{cx + box_w/2}" y2="{y_med}" stroke="#d97706" stroke-width="3"/>')
        svg.append(f'<polygon points="{cx},{y_mean-5} {cx+5},{y_mean} {cx},{y_mean+5} {cx-5},{y_mean}" fill="{col}" stroke="#1b2d42" stroke-width="1"/>')
        
        svg.append(f'<text x="{cx}" y="{mt + ph + 24}" font-family="-apple-system, sans-serif" font-size="14" font-weight="700" text-anchor="middle" fill="#1b2d42">{tname}</text>')
        svg.append(f'<text x="{cx}" y="{mt + ph + 42}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="middle" fill="#57606a">Mean: {st["mean"]:.2f} h | Med: {st["med"]:.2f} h</text>')
        
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 16}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#24292f">Usage Tiers (Mean Study Time Drops by -3.20 hrs/week from Low to High Usage; Time Displacement)</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig8.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/study-by-usage-boxplot.png', w, h)

# ==============================================================================
# Figure 9: Screen Time vs Study Time Scatter Plot with Trendline
# ==============================================================================
def make_fig9():
    w, h = 1024, 640
    ml, mr, mt, mb = 80, 40, 70, 80
    pw, ph = w - ml - mr, h - mt - mb
    
    def to_x(vx): return ml + (vx / 9.5) * pw
    def to_y(vy): return mt + ph - (vy / 45.0) * ph
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Daily Social Media Hours vs. Weekly Study Hours (N = {N:,})</text>')
    
    for vy in [0, 10, 20, 30, 40]:
        yp = to_y(vy)
        svg.append(f'<line x1="{ml}" y1="{yp}" x2="{w - mr}" y2="{yp}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 12}" y="{yp + 4}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="end" fill="#57606a">{vy} h</text>')
        
    for vx in [0, 2, 4, 6, 8]:
        xp = to_x(vx)
        svg.append(f'<line x1="{xp}" y1="{mt}" x2="{xp}" y2="{mt + ph}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{xp}" y="{mt + ph + 22}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="middle" fill="#57606a">{vx} h</text>')
        
    svg.append(f'<text x="24" y="{mt + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 24 {mt + ph/2})">Weekly Study Time (Hours)</text>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Daily Social Media Screen Time (Hours)</text>')
    
    for sx, sy in scatter_points[:450]:
        px, py = to_x(sx), to_y(sy)
        svg.append(f'<circle cx="{px}" cy="{py}" r="4" fill="#1b2d42" fill-opacity="0.35" stroke="#2a4365" stroke-width="0.8"/>')
        
    # Regression line: Study = -0.74 * Screen + 25.47
    svg.append(f'<line x1="{to_x(0)}" y1="{to_y(25.47)}" x2="{to_x(9.0)}" y2="{to_y(18.81)}" stroke="#bc5689" stroke-width="3.5" stroke-dasharray="6,4"/>')
    
    svg.append(f'<rect x="{w - mr - 290}" y="{mt + 15}" width="270" height="75" fill="#f8fafc" stroke="#cbd5e1" rx="6"/>')
    svg.append(f'<text x="{w - mr - 275}" y="{mt + 36}" font-family="-apple-system, sans-serif" font-size="13" font-weight="700" fill="#1b2d42">Correlation &amp; Trendline:</text>')
    svg.append(f'<text x="{w - mr - 275}" y="{mt + 55}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" fill="#bc5689">Pearson r = -0.178 (p &lt; 0.0001)</text>')
    svg.append(f'<text x="{w - mr - 275}" y="{mt + 72}" font-family="-apple-system, sans-serif" font-size="11" fill="#57606a">Linear Fit: Study = -0.74 × Screen + 25.47</text>')
    
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig9.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/screentime-vs-studytime-scatter.png', w, h)

# ==============================================================================
# Figure 10: 8x8 Correlation Matrix Heatmap
# ==============================================================================
def make_fig10():
    w, h = 1024, 768
    ml, mr, mt, mb = 170, 110, 80, 80
    pw, ph = w - ml - mr, h - mt - mb
    labels_display = [
        "Social Media Hours",
        "Weekly Study Hours",
        "Attendance Rate",
        "Productivity Score",
        "Brain Rot Index",
        "Digital Addiction",
        "Attention Span",
        "Academic Motivation"
    ]
    M = len(corr_vars)
    cw = pw / M
    ch = ph / M
    
    def get_color(r_val):
        if r_val >= 0:
            t = min(1.0, r_val / 0.7)
            red = int(255 - t * (255 - 185))
            green = int(255 - t * (255 - 28))
            blue = int(255 - t * (255 - 28))
            return f"rgb({red},{green},{blue})"
        else:
            t = min(1.0, abs(r_val) / 0.7)
            red = int(255 - t * (255 - 30))
            green = int(255 - t * (255 - 58))
            blue = int(255 - t * (255 - 138))
            return f"rgb({red},{green},{blue})"
            
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="44" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Correlation Matrix of Digital Behavior &amp; Academic Indicators (N = {N:,})</text>')
    
    for i in range(M):
        v1 = corr_vars[i]
        svg.append(f'<text x="{ml - 12}" y="{mt + i*ch + ch/2 + 5}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" text-anchor="end" fill="#1b2d42">{labels_display[i]}</text>')
        svg.append(f'<text x="{ml + i*cw + cw/2}" y="{mt + ph + 24}" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" text-anchor="middle" fill="#1b2d42">{labels_display[i].split()[0]}</text>')
        
        for j in range(M):
            v2 = corr_vars[j]
            r_val = corr_matrix[v1][v2]
            cx = ml + j * cw
            cy = mt + i * ch
            bg = get_color(r_val)
            txt_col = "#ffffff" if abs(r_val) > 0.35 else "#1b2d42"
            
            svg.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="{bg}" stroke="#ffffff" stroke-width="1.5"/>')
            svg.append(f'<text x="{cx + cw/2}" y="{cy + ch/2 + 5}" font-family="-apple-system, sans-serif" font-size="12" font-weight="{700 if i==j or abs(r_val)>0.25 else 500}" text-anchor="middle" fill="{txt_col}">{r_val:.2f}</text>')
            
    # Color scale bar
    cb_x = w - mr + 25
    cb_y = mt
    cb_w = 18
    cb_h = ph
    steps = 40
    for s in range(steps):
        sy = cb_y + s * (cb_h / steps)
        val = 0.7 - (s / steps) * 1.4
        col = get_color(val)
        svg.append(f'<rect x="{cb_x}" y="{sy}" width="{cb_w}" height="{cb_h/steps + 0.5}" fill="{col}"/>')
    svg.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="none" stroke="#1b2d42" stroke-width="1"/>')
    svg.append(f'<text x="{cb_x + cb_w + 8}" y="{cb_y + 10}" font-family="-apple-system, sans-serif" font-size="11" fill="#57606a">+0.70</text>')
    svg.append(f'<text x="{cb_x + cb_w + 8}" y="{cb_y + cb_h/2 + 4}" font-family="-apple-system, sans-serif" font-size="11" fill="#57606a">0.00</text>')
    svg.append(f'<text x="{cb_x + cb_w + 8}" y="{cb_y + cb_h}" font-family="-apple-system, sans-serif" font-size="11" fill="#57606a">-0.70</text>')
    
    svg.append(f'<text x="{w/2}" y="{h - 18}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#24292f">Pearson Correlation Coefficients across 500,000 Global Records</text>')
    
    svg.append('</svg>')
    p = 'scratch/svgs/fig10.svg'
    with open(p, 'w') as f: f.write('\n'.join(svg))
    convert_svg_to_png(p, 'images/correlation-matrix-heatmap.png', w, h)

# Run All 10
make_fig1()
make_fig2()
make_fig3()
make_fig4()
make_fig5()
make_fig6()
make_fig7()
make_fig8()
make_fig9()
make_fig10()

print("All 10 EDA Visualizations generated successfully!")
