# Social Media Usage and Student Well-Being

A semester-long academic research project investigating how social media usage associates with students' mental health, physical health, sleep, academic performance, and daily habits.

## 📋 Research Overview

- **Topic**: The impact of social media usage on students.
- **Core Research Question**: *How is social media usage associated with students' mental and physical health and daily habits?*
- **Primary Data Source**: Secondary student survey dataset from Kaggle.
- **Current Milestone**: Phase 2 — Dataset Selection & Local Architecture.

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
- **H1**: Students who spend more time on social media may be more likely to use it as a distraction from personal problems, which could interfere with their ability to focus on schoolwork and daily responsibilities.
- **H2**: Students who are easily distracted by social media may experience negative effects on their academic performance and sleep.

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
├── dataset.html      # Kaggle Dataset details, variable dictionary, and methodology
├── analysis.html     # Exploratory Data Analysis visualizations (histograms, bar charts, box plots, heatmaps)
├── findings.html     # Hypotheses evaluation status (H1 & H2) and study phase
├── resources.html    # Presentation video embed, proposal documents, and references
├── resume.html       # Standalone printable resume page for Jimena Bello
├── css/
│   └── style.css     # CSS design tokens, responsive layout, and visual styles
├── js/
│   └── main.js       # Vanilla JS for multi-page active links, mobile menu, & topic filters
├── .gitignore        # Ignores macOS system files and IDE caches
└── README.md         # Academic documentation and local setup instructions
```
