# Social Media Usage and Student Well-Being

A semester-long academic research project investigating how social media usage associates with students' mental health, physical health, sleep, academic performance, and daily habits.

## 📋 Research Overview

- **Topic**: The impact of social media usage on students.
- **Core Research Question**: *How is social media usage associated with students' mental and physical health and daily habits?*
- **Primary Data Source**: [Nitika Chandel: Social Media Impact on Students' Behavior & Outcome](https://www.kaggle.com/code/nitikachandel95/social-media-impact-on-students-behavior-outcome) (*Global Student Digital Behavior Dataset*, $N = 500,000$ records across 50 countries).
- **Current Milestone**: Phase 4 — Relational SQL Modeling & Machine Learning Evaluation.

---

## 🔬 Research Framework

### Sub-Questions
1. How does time spent on social media affect students' sleep time and sleep comfort?
2. Does social media usage affect students' academic performance?
3. Does social media usage have different impacts on students based on individual traits?
4. Are certain social media platforms used more frequently than others?
5. Is heavier social media usage associated with lower mental or physical well-being?

### Evaluated Variables
- Daily social media usage
- Age
- Gender
- Social media platforms used
- Sleep problems
- Academic performance
- Overall well-being
- Stress and pressure

### Hypotheses
- **H1**: Students who spend more time on social media experience reduced weekly study hours due to direct time competition with academic responsibilities (Supported by empirical regression & SQL analysis).
- **H2**: Frequent social media interruptions and elevated digital addiction degrade sustained attention stamina, sleep quality, and academic productivity (Supported by empirical clustering & ML feature importance).

---

## 💻 Tech Stack & Design Decisions

- **Pure Vanilla Web Architecture**: Built strictly with **HTML5**, **CSS3**, and **vanilla JavaScript**.
- **No Build Step / No Dependencies**: Zero `npm` packages or heavy frameworks, ensuring ultra-fast load times, maximum accessibility, and instant deployment.
- **Deployment Ready**: Fully configured for zero-configuration static deployment on **GitHub Pages** or **Vercel**.

---

## 🚀 How to Run Locally (Preview Mode)

Because this website uses pure HTML, CSS, and JavaScript, you can run a local preview server with Python:

1. Open your terminal and navigate to the project directory:
   ```bash
   cd /Users/jime/social-media-research
   ```

2. Start a lightweight local HTTP server:
   ```bash
   python3 -m http.server 8000
   ```

3. Open your web browser and visit:
   ```
   http://localhost:8000
   ```

---

## 📂 Project Structure

```
social-media-research/
├── index.html        # Home page (Executive Summary, Core Question, Portal Cards, About)
├── research.html     # Research Framework (Topic Pillars, 5 Sub-Questions, 8 Variables, Hypotheses)
├── dataset.html      # Kaggle Dataset details, sample summary (N=500,000), and cleaning pipeline
├── analysis.html     # Exploratory Data Analysis (10 charts) & SQL relational analysis (8 queries)
├── findings.html     # Hypothesis validation status, Machine Learning models (OLS & Random Forest), Limitations
├── resources.html    # Presentation video embed, proposal documents, and Kaggle references
├── resume.html       # Standalone printable resume page for Jimena Bello
├── data/
│   └── global_student_digital_behavior_dataset.csv # Global dataset (N=500,000, 304 MB, local only / gitignored)
├── scripts/
│   ├── clean_global_dataset.py         # Comprehensive data audit & quality checks (48 columns)
│   ├── generate_eda_visualizations_v2.py # 10 empirical EDA visualization generator
│   ├── train_ml_models.py              # Dual-target ML pipeline (OLS & Random Forest Regressor/Classifier)
│   ├── calculate_sql_results.py        # Verification script for 8 core SQL queries
│   └── analysis_queries.sql            # SQL schema & analytical query definitions
├── images/           # High-resolution PNG figures (10 EDA charts + 4 ML evaluation plots)
├── css/
│   └── style.css     # CSS design tokens, responsive layout, and visual styles
├── js/
│   └── main.js       # Vanilla JS for multi-page active links, mobile menu, & topic filters
├── .gitignore        # Ignores macOS system files, IDE caches, and 304 MB CSV dataset
└── README.md         # Academic documentation and local setup instructions
```
