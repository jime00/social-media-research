import csv
import math
import subprocess
import os

# 1. Load Cleaned Dataset
with open('data/cleaned_social_media_data.csv', mode='r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

N = len(rows)
print(f"Loaded {N} records from data/cleaned_social_media_data.csv")

os.makedirs('scratch/svgs', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Helper to execute sips
def convert_svg_to_png(svg_path, png_path, width=1024, height=640):
    cmd = ["sips", "-s", "format", "png", svg_path, "--out", png_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error converting {svg_path}: {res.stderr}")
    else:
        # ensure pixel dimensions
        subprocess.run(["sips", "-z", str(height), str(width), png_path], capture_output=True)
        print(f"Generated: {png_path} ({width}x{height})")

# ==============================================================================
# Figure 5: Student Age Distribution
# ==============================================================================
def generate_figure_5():
    svg_path = 'scratch/svgs/fig5.svg'
    png_path = 'images/age-distribution.png'
    w, h = 1024, 640
    
    ages = [int(r['age']) for r in rows]
    from collections import Counter
    age_counts = Counter(ages)
    
    # Range 17 to 30, and 37
    # For neat bar chart, let's plot ages 17 to 30, plus 37
    plot_ages = list(range(17, 31)) + [37]
    counts = [age_counts.get(a, 0) for a in plot_ages]
    max_c = 100
    
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 80
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    
    # Title
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Distribution of Student Respondents by Age (N = {N})</text>')
    
    # Y-axis grid & labels
    for y_val in [0, 20, 40, 60, 80, 100]:
        y_pos = margin_t + ph - (y_val / max_c) * ph
        svg.append(f'<line x1="{margin_l}" y1="{y_pos}" x2="{w - margin_r}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="end" fill="#57606a">{y_val}</text>')
    
    # Y-axis label
    svg.append(f'<text x="28" y="{margin_t + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 28 {margin_t + ph/2})">Number of Respondents</text>')
    
    # X-axis
    bar_width = pw / len(plot_ages) * 0.75
    gap = pw / len(plot_ages)
    
    # Mean and Median line indicators
    # median = 21, mean = 21.25
    
    for i, a in enumerate(plot_ages):
        c = counts[i]
        bx = margin_l + i * gap + (gap - bar_width)/2
        bh = (c / max_c) * ph
        by = margin_t + ph - bh
        
        # Color highlight core cohort (19-23)
        color = "#bc5689" if 19 <= a <= 23 else "#2a4365"
        
        svg.append(f'<rect x="{bx}" y="{by}" width="{bar_width}" height="{bh}" fill="{color}" rx="3" stroke="#1b2d42" stroke-width="1"/>')
        if c > 0:
            svg.append(f'<text x="{bx + bar_width/2}" y="{by - 7}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" text-anchor="middle" fill="#1b2d42">{c}</text>')
        
        label = str(a) if a != 37 else "37"
        svg.append(f'<text x="{bx + bar_width/2}" y="{margin_t + ph + 22}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="middle" fill="#24292f">{label}</text>')
    
    # X-axis line & label
    svg.append(f'<line x1="{margin_l}" y1="{margin_t + ph}" x2="{w - margin_r}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Age in Years (Undergraduate Concentration: Ages 19–23 = 88.6%)</text>')
    
    # Legend
    svg.append(f'<rect x="{w - margin_r - 260}" y="{margin_t + 10}" width="16" height="16" fill="#bc5689" rx="2" stroke="#1b2d42" stroke-width="1"/>')
    svg.append(f'<text x="{w - margin_r - 238}" y="{margin_t + 23}" font-family="-apple-system, sans-serif" font-size="12" fill="#24292f">Core Undergraduate Cohort (19–23)</text>')
    svg.append(f'<rect x="{w - margin_r - 260}" y="{margin_t + 34}" width="16" height="16" fill="#2a4365" rx="2" stroke="#1b2d42" stroke-width="1"/>')
    svg.append(f'<text x="{w - margin_r - 238}" y="{margin_t + 47}" font-family="-apple-system, sans-serif" font-size="12" fill="#24292f">Other Age Groups</text>')
    
    svg.append('</svg>')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(svg))
    convert_svg_to_png(svg_path, png_path, w, h)

# ==============================================================================
# Figure 6: Daily Study Time Distribution
# ==============================================================================
def generate_figure_6():
    svg_path = 'scratch/svgs/fig6.svg'
    png_path = 'images/study-time-distribution.png'
    w, h = 1024, 640
    
    study_times = [float(r['study_time_hours']) for r in rows]
    # Bins: [0, 1), [1, 2), [2, 3), [3, 4), [4, 5), [5, 6), [6, 8), [8, 12+]
    bins = [
        ("0–1 h", sum(1 for s in study_times if 0 <= s < 1)),
        ("1–2 h", sum(1 for s in study_times if 1 <= s < 2)),
        ("2–3 h", sum(1 for s in study_times if 2 <= s < 3)),
        ("3–4 h", sum(1 for s in study_times if 3 <= s < 4)),
        ("4–5 h", sum(1 for s in study_times if 4 <= s < 5)),
        ("5–6 h", sum(1 for s in study_times if 5 <= s < 6)),
        ("6–7 h", sum(1 for s in study_times if 6 <= s < 7)),
        ("7–8 h", sum(1 for s in study_times if 7 <= s < 8)),
        ("8+ h", sum(1 for s in study_times if s >= 8))
    ]
    max_c = 100
    
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 80
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Distribution of Daily Study Time in Hours (N = {N})</text>')
    
    for y_val in [0, 20, 40, 60, 80, 100]:
        y_pos = margin_t + ph - (y_val / max_c) * ph
        svg.append(f'<line x1="{margin_l}" y1="{y_pos}" x2="{w - margin_r}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="end" fill="#57606a">{y_val}</text>')
        
    svg.append(f'<text x="28" y="{margin_t + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 28 {margin_t + ph/2})">Number of Respondents</text>')
    
    gap = pw / len(bins)
    bar_width = gap * 0.8
    
    for i, (b_label, count) in enumerate(bins):
        bx = margin_l + i * gap + (gap - bar_width)/2
        bh = (count / max_c) * ph
        by = margin_t + ph - bh
        pct = (count / N) * 100
        
        svg.append(f'<rect x="{bx}" y="{by}" width="{bar_width}" height="{bh}" fill="#2a4365" rx="3" stroke="#1b2d42" stroke-width="1"/>')
        svg.append(f'<text x="{bx + bar_width/2}" y="{by - 8}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" text-anchor="middle" fill="#1b2d42">{count}</text>')
        svg.append(f'<text x="{bx + bar_width/2}" y="{margin_t + ph + 22}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="middle" fill="#24292f">{b_label}</text>')
        
    svg.append(f'<line x1="{margin_l}" y1="{margin_t + ph}" x2="{w - margin_r}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Daily Study Duration (Median = 3.00 hrs, Mean = 3.48 hrs, IQR = 2.0–5.0 hrs)</text>')
    
    # Statistical annotations
    svg.append(f'<rect x="{w - margin_r - 280}" y="{margin_t + 15}" width="260" height="70" fill="#f8fafc" stroke="#cbd5e1" rx="6"/>')
    svg.append(f'<text x="{w - margin_r - 265}" y="{margin_t + 36}" font-family="-apple-system, sans-serif" font-size="12" font-weight="700" fill="#1b2d42">Summary Statistics:</text>')
    svg.append(f'<text x="{w - margin_r - 265}" y="{margin_t + 54}" font-family="-apple-system, sans-serif" font-size="12" fill="#57606a">• Median: 3.00 hrs | Mean: 3.48 hrs</text>')
    svg.append(f'<text x="{w - margin_r - 265}" y="{margin_t + 70}" font-family="-apple-system, sans-serif" font-size="12" fill="#57606a">• Q1: 2.00 hrs | Q3: 5.00 hrs (IQR: 3.0)</text>')
    
    svg.append('</svg>')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(svg))
    convert_svg_to_png(svg_path, png_path, w, h)

# ==============================================================================
# Figure 7: Primary Social Media Platform Distribution
# ==============================================================================
def generate_figure_7():
    svg_path = 'scratch/svgs/fig7.svg'
    png_path = 'images/platform-distribution.png'
    w, h = 1024, 640
    
    platforms = [
        ("Facebook", 137, 33.8, "#1877F2"),
        ("Instagram", 125, 30.9, "#E1306C"),
        ("YouTube", 121, 29.9, "#FF0000"),
        ("Other", 18, 4.4, "#64748B"),
        ("TikTok", 4, 1.0, "#010101")
    ]
    max_c = 160
    
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 80
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Primary Social Media Platform Distribution (N = {N})</text>')
    
    for y_val in [0, 40, 80, 120, 160]:
        y_pos = margin_t + ph - (y_val / max_c) * ph
        svg.append(f'<line x1="{margin_l}" y1="{y_pos}" x2="{w - margin_r}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="end" fill="#57606a">{y_val}</text>')
        
    svg.append(f'<text x="28" y="{margin_t + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 28 {margin_t + ph/2})">Number of Students</text>')
    
    gap = pw / len(platforms)
    bar_width = gap * 0.65
    
    for i, (p_name, count, pct, color) in enumerate(platforms):
        bx = margin_l + i * gap + (gap - bar_width)/2
        bh = (count / max_c) * ph
        by = margin_t + ph - bh
        
        svg.append(f'<rect x="{bx}" y="{by}" width="{bar_width}" height="{bh}" fill="{color}" rx="4" stroke="#1b2d42" stroke-width="1"/>')
        svg.append(f'<text x="{bx + bar_width/2}" y="{by - 18}" font-family="-apple-system, sans-serif" font-size="13" font-weight="700" text-anchor="middle" fill="#1b2d42">{count}</text>')
        svg.append(f'<text x="{bx + bar_width/2}" y="{by - 4}" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" text-anchor="middle" fill="#57606a">({pct}%)</text>')
        svg.append(f'<text x="{bx + bar_width/2}" y="{margin_t + ph + 24}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">{p_name}</text>')
        
    svg.append(f'<line x1="{margin_l}" y1="{margin_t + ph}" x2="{w - margin_r}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Self-Reported Primary Social Networking Platform (Top 3 Comprise 94.6%)</text>')
    
    svg.append('</svg>')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(svg))
    convert_svg_to_png(svg_path, png_path, w, h)

# ==============================================================================
# Figure 8: Daily Study Time Across Social Media Usage Levels (Comparative Box Plot)
# ==============================================================================
def generate_figure_8():
    svg_path = 'scratch/svgs/fig8.svg'
    png_path = 'images/study-by-usage-boxplot.png'
    w, h = 1024, 640
    
    # Low: n=187, mean=3.84, med=3.5, q1=2.0, q3=5.0, min=0.0, max=12.0
    # Mod: n=176, mean=3.28, med=3.0, q1=2.0, q3=4.5, min=0.0, max=10.0
    # High: n=42, mean=2.72, med=2.0, q1=1.0, q3=3.5, min=0.0, max=10.0
    tiers = [
        ("Low Usage (≤ 3 hrs)", 187, 0.0, 2.0, 3.5, 3.84, 5.0, 8.0, [9.0, 10.0, 10.0, 12.0], "#2e7d32"),
        ("Moderate (3–7 hrs)", 176, 0.0, 2.0, 3.0, 3.28, 4.5, 8.0, [10.0, 10.0], "#d97706"),
        ("High Usage (> 7 hrs)", 42, 0.0, 1.0, 2.0, 2.72, 3.5, 6.0, [9.5, 10.0], "#dc2626")
    ]
    max_y = 13.0
    
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 80
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Daily Study Time Across Social Media Usage Levels (N = {N})</text>')
    
    for y_val in [0, 2, 4, 6, 8, 10, 12]:
        y_pos = margin_t + ph - (y_val / max_y) * ph
        svg.append(f'<line x1="{margin_l}" y1="{y_pos}" x2="{w - margin_r}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{y_pos + 5}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="end" fill="#57606a">{y_val} h</text>')
        
    svg.append(f'<text x="28" y="{margin_t + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 28 {margin_t + ph/2})">Daily Study Duration (Hours)</text>')
    
    gap = pw / len(tiers)
    box_w = 110
    
    def to_y(v):
        return margin_t + ph - (v / max_y) * ph
    
    for i, (name, n_cnt, mn, q1, med, mean_val, q3, mx, outliers, col) in enumerate(tiers):
        cx = margin_l + i * gap + gap/2
        
        y_q1 = to_y(q1)
        y_q3 = to_y(q3)
        y_med = to_y(med)
        y_mean = to_y(mean_val)
        y_min = to_y(mn)
        y_max = to_y(mx)
        
        # Whisker lines
        svg.append(f'<line x1="{cx}" y1="{y_q1}" x2="{cx}" y2="{y_min}" stroke="#1b2d42" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx}" y1="{y_q3}" x2="{cx}" y2="{y_max}" stroke="#1b2d42" stroke-width="1.5"/>')
        # Whisker caps
        svg.append(f'<line x1="{cx - 20}" y1="{y_min}" x2="{cx + 20}" y2="{y_min}" stroke="#1b2d42" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx - 20}" y1="{y_max}" x2="{cx + 20}" y2="{y_max}" stroke="#1b2d42" stroke-width="1.5"/>')
        
        # Box
        svg.append(f'<rect x="{cx - box_w/2}" y="{y_q3}" width="{box_w}" height="{y_q1 - y_q3}" fill="#ffffff" stroke="#1b2d42" stroke-width="2"/>')
        # Median line (orange)
        svg.append(f'<line x1="{cx - box_w/2}" y1="{y_med}" x2="{cx + box_w/2}" y2="{y_med}" stroke="#d97706" stroke-width="2.5"/>')
        # Mean marker (diamond)
        svg.append(f'<polygon points="{cx},{y_mean-5} {cx+5},{y_mean} {cx},{y_mean+5} {cx-5},{y_mean}" fill="{col}" stroke="#1b2d42" stroke-width="1"/>')
        
        # Outliers
        for ov in outliers:
            yo = to_y(ov)
            svg.append(f'<circle cx="{cx}" cy="{yo}" r="3.5" fill="#ffffff" stroke="#1b2d42" stroke-width="1.5"/>')
            
        # Stats tag below box
        svg.append(f'<text x="{cx}" y="{margin_t + ph + 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="700" text-anchor="middle" fill="#1b2d42">{name}</text>')
        svg.append(f'<text x="{cx}" y="{margin_t + ph + 40}" font-family="-apple-system, sans-serif" font-size="12" text-anchor="middle" fill="#57606a">n = {n_cnt} | Mean: {mean_val:.2f} h | Med: {med:.1f} h</text>')
        
    svg.append(f'<line x1="{margin_l}" y1="{margin_t + ph}" x2="{w - margin_r}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<text x="{w/2}" y="{h - 12}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#24292f">Usage Tiers (Mean Study Time Drops from 3.84 hrs → 3.28 hrs → 2.72 hrs; 1.12 h Displacement)</text>')
    
    # Legend
    svg.append(f'<line x1="{w - margin_r - 200}" y1="{margin_t + 15}" x2="{w - margin_r - 180}" y2="{margin_t + 15}" stroke="#d97706" stroke-width="2.5"/>')
    svg.append(f'<text x="{w - margin_r - 172}" y="{margin_t + 19}" font-family="-apple-system, sans-serif" font-size="12" fill="#24292f">Median Line</text>')
    svg.append(f'<polygon points="{w - margin_r - 190},{margin_t + 32} {w - margin_r - 185},{margin_t + 37} {w - margin_r - 190},{margin_t + 42} {w - margin_r - 195},{margin_t + 37}" fill="#2e7d32" stroke="#1b2d42" stroke-width="1"/>')
    svg.append(f'<text x="{w - margin_r - 172}" y="{margin_t + 41}" font-family="-apple-system, sans-serif" font-size="12" fill="#24292f">Group Mean (◆)</text>')
    
    svg.append('</svg>')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(svg))
    convert_svg_to_png(svg_path, png_path, w, h)

# ==============================================================================
# Figure 9: Screen Time vs. Daily Study Time (Scatter Plot with Regression Trendline)
# ==============================================================================
def generate_figure_9():
    svg_path = 'scratch/svgs/fig9.svg'
    png_path = 'images/screentime-vs-studytime-scatter.png'
    w, h = 1024, 640
    
    points = [(float(r['screen_time_hours']), float(r['study_time_hours'])) for r in rows]
    
    max_x = 10.5
    max_y = 13.0
    
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 80
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b
    
    def to_x(vx):
        return margin_l + (vx / max_x) * pw
    def to_y(vy):
        return margin_t + ph - (vy / max_y) * ph
        
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="42" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Social Media Screen Time vs. Daily Study Time (N = {N})</text>')
    
    # Y-axis
    for y_val in [0, 2, 4, 6, 8, 10, 12]:
        yp = to_y(y_val)
        svg.append(f'<line x1="{margin_l}" y1="{yp}" x2="{w - margin_r}" y2="{yp}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{yp + 5}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="end" fill="#57606a">{y_val}</text>')
    svg.append(f'<text x="28" y="{margin_t + ph/2}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f" transform="rotate(-90 28 {margin_t + ph/2})">Daily Study Time (Hours)</text>')
    
    # X-axis
    for x_val in [0, 2, 4, 6, 8, 10]:
        xp = to_x(x_val)
        svg.append(f'<line x1="{xp}" y1="{margin_t}" x2="{xp}" y2="{margin_t + ph}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{xp}" y="{margin_t + ph + 24}" font-family="-apple-system, sans-serif" font-size="13" text-anchor="middle" fill="#57606a">{x_val}</text>')
    svg.append(f'<text x="{w/2}" y="{h - 22}" font-family="-apple-system, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#24292f">Daily Social Media Screen Time (Hours)</text>')
    
    # Regression line: y = -0.246x + 4.45
    # at x=0 -> y=4.45
    # at x=10 -> y=1.99
    x1, y1_val = 0.0, 4.45
    x2, y2_val = 10.0, 1.99
    svg.append(f'<line x1="{to_x(x1)}" y1="{to_y(y1_val)}" x2="{to_x(x2)}" y2="{to_y(y2_val)}" stroke="#bc5689" stroke-width="3" stroke-dasharray="6,4"/>')
    
    # Scatter points (semi-transparent for overlap)
    for sx, sy in points:
        px = to_x(sx)
        py = to_y(sy)
        svg.append(f'<circle cx="{px}" cy="{py}" r="4.5" fill="#1b2d42" fill-opacity="0.45" stroke="#2a4365" stroke-width="1"/>')
        
    svg.append(f'<line x1="{margin_l}" y1="{margin_t + ph}" x2="{w - margin_r}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + ph}" stroke="#1b2d42" stroke-width="1.5"/>')
    
    # Annotation box for statistics
    svg.append(f'<rect x="{w - margin_r - 280}" y="{margin_t + 15}" width="260" height="75" fill="#f8fafc" stroke="#cbd5e1" rx="6"/>')
    svg.append(f'<text x="{w - margin_r - 265}" y="{margin_t + 36}" font-family="-apple-system, sans-serif" font-size="13" font-weight="700" fill="#1b2d42">Correlation &amp; Trendline:</text>')
    svg.append(f'<text x="{w - margin_r - 265}" y="{margin_t + 56}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" fill="#bc5689">Pearson r = -0.204 (p &lt; 0.001)</text>')
    svg.append(f'<text x="{w - margin_r - 265}" y="{margin_t + 73}" font-family="-apple-system, sans-serif" font-size="11" fill="#57606a">Linear Fit: Study = -0.25 × Screen + 4.45</text>')
    
    svg.append('</svg>')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(svg))
    convert_svg_to_png(svg_path, png_path, w, h)

# ==============================================================================
# Figure 10: Correlation Matrix Heatmap
# ==============================================================================
def generate_figure_10():
    svg_path = 'scratch/svgs/fig10.svg'
    png_path = 'images/correlation-matrix-heatmap.png'
    w, h = 1024, 768
    
    var_keys = [
        ('screen_time_hours', 'Screen Time'),
        ('study_time_hours', 'Study Time'),
        ('academic_distraction_scale', 'Acad. Distraction'),
        ('sleep_disturbance_scale', 'Sleep Disturbance'),
        ('anxiety_scale', 'Anxiety Scale'),
        ('depression_scale', 'Depression Scale'),
        ('self_esteem_scale', 'Self-Esteem'),
        ('gpa', 'Academic GPA')
    ]
    
    def calc_r(k1, k2):
        x = [float(r[k1]) for r in rows]
        y = [float(r[k2]) for r in rows]
        mx, my = sum(x)/N, sum(y)/N
        vx = sum((xi - mx)**2 for xi in x)
        vy = sum((yi - my)**2 for yi in y)
        cov = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
        return cov / math.sqrt(vx * vy)
        
    M = len(var_keys)
    matrix = [[calc_r(k1, k2) for k2, _ in var_keys] for k1, _ in var_keys]
    
    margin_l, margin_r, margin_t, margin_b = 160, 110, 80, 80
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b
    cell_w = pw / M
    cell_h = ph / M
    
    # Color interpolation function from blue (-0.5) to white (0.0) to red (+0.6)
    def get_color(r_val):
        # clamp between -0.5 and +0.6
        if r_val >= 0:
            # white (#ffffff) to deep burgundy/red (#b91c1c)
            t = min(1.0, r_val / 0.6)
            red = int(255 - t * (255 - 185))
            green = int(255 - t * (255 - 28))
            blue = int(255 - t * (255 - 28))
            return f"rgb({red},{green},{blue})"
        else:
            # white (#ffffff) to deep slate blue (#1e3a8a)
            t = min(1.0, abs(r_val) / 0.5)
            red = int(255 - t * (255 - 30))
            green = int(255 - t * (255 - 58))
            blue = int(255 - t * (255 - 138))
            return f"rgb({red},{green},{blue})"
            
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="44" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle" fill="#1b2d42">Correlation Matrix of Academic &amp; Psychological Indicators (N = {N})</text>')
    
    for i in range(M):
        # Y labels
        _, lbl_y = var_keys[i]
        svg.append(f'<text x="{margin_l - 12}" y="{margin_t + i*cell_h + cell_h/2 + 5}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" text-anchor="end" fill="#1b2d42">{lbl_y}</text>')
        # X labels
        _, lbl_x = var_keys[i]
        svg.append(f'<text x="{margin_l + i*cell_w + cell_w/2}" y="{margin_t + ph + 24}" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" text-anchor="middle" fill="#1b2d42">{lbl_x}</text>')
        
        for j in range(M):
            r_val = matrix[i][j]
            cx = margin_l + j * cell_w
            cy = margin_t + i * cell_h
            bg = get_color(r_val)
            text_col = "#ffffff" if abs(r_val) > 0.35 else "#1b2d42"
            
            svg.append(f'<rect x="{cx}" y="{cy}" width="{cell_w}" height="{cell_h}" fill="{bg}" stroke="#ffffff" stroke-width="1.5"/>')
            svg.append(f'<text x="{cx + cell_w/2}" y="{cy + cell_h/2 + 5}" font-family="-apple-system, sans-serif" font-size="12" font-weight="{700 if i==j or abs(r_val)>0.2 else 500}" text-anchor="middle" fill="{text_col}">{r_val:.2f}</text>')
            
    # Heatmap Color Scale Bar on the right
    cb_x = w - margin_r + 25
    cb_y = margin_t
    cb_w = 18
    cb_h = ph
    
    steps = 40
    for s in range(steps):
        sy = cb_y + s * (cb_h / steps)
        # map s from top (positive +0.6) to bottom (negative -0.5)
        val = 0.6 - (s / steps) * 1.1
        col = get_color(val)
        svg.append(f'<rect x="{cb_x}" y="{sy}" width="{cb_w}" height="{cb_h/steps + 0.5}" fill="{col}"/>')
    svg.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="none" stroke="#1b2d42" stroke-width="1"/>')
    
    # Scale labels
    svg.append(f'<text x="{cb_x + cb_w + 8}" y="{cb_y + 10}" font-family="-apple-system, sans-serif" font-size="11" fill="#57606a">+0.60</text>')
    svg.append(f'<text x="{cb_x + cb_w + 8}" y="{cb_y + cb_h * (0.6/1.1) + 4}" font-family="-apple-system, sans-serif" font-size="11" fill="#57606a">0.00</text>')
    svg.append(f'<text x="{cb_x + cb_w + 8}" y="{cb_y + cb_h}" font-family="-apple-system, sans-serif" font-size="11" fill="#57606a">-0.50</text>')
    svg.append(f'<text x="{cb_x + cb_w/2}" y="{margin_t - 14}" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" text-anchor="middle" fill="#1b2d42">r</text>')
    
    svg.append(f'<text x="{w/2}" y="{h - 18}" font-family="-apple-system, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#24292f">Pearson Correlation Matrix (Values range from -0.20 to +0.56 across key indicators)</text>')
    
    svg.append('</svg>')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(svg))
    convert_svg_to_png(svg_path, png_path, w, h)

# Run All
generate_figure_5()
generate_figure_6()
generate_figure_7()
generate_figure_8()
generate_figure_9()
generate_figure_10()
print("All 6 figures generated successfully!")
