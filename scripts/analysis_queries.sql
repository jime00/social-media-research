-- ============================================================================
-- Assignment 4: SQL Relational Data Analysis
-- Course Project: Student Social Media Usage & Academic Performance
-- Database Schema & Analytical Queries (Global Dataset: N = 500,000)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. TABLE CREATION: Countries (Lookup / Dimension Table)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS countries;

CREATE TABLE countries (
    country_id INT PRIMARY KEY,
    country_name VARCHAR(60) NOT NULL UNIQUE,
    development_level VARCHAR(30) NOT NULL,
    poverty_rate_percent DECIMAL(5,2),
    internet_infrastructure_index DECIMAL(5,2)
);

-- Seed Data: Sample Country Information (Representative 50 Countries)
INSERT INTO countries (country_id, country_name, development_level, poverty_rate_percent, internet_infrastructure_index) VALUES
(1, 'United States', 'Developed', 11.50, 88.20),
(2, 'United Kingdom', 'Developed', 12.00, 86.50),
(3, 'India', 'Developing', 18.20, 62.40),
(4, 'China', 'Developing', 14.10, 78.90),
(5, 'Kenya', 'Underdeveloped', 36.10, 38.50),
(6, 'Cambodia', 'Developing', 17.80, 48.20);

-- ----------------------------------------------------------------------------
-- 2. TABLE CREATION: Students (Main Entity / Fact Table)
-- ----------------------------------------------------------------------------
CREATE TABLE students (
    student_id INT PRIMARY KEY,
    country_id INT,
    age INT,
    gender VARCHAR(15),
    device_access VARCHAR(30),
    social_media_hours DECIMAL(4,2),
    study_hours_per_week DECIMAL(5,2),
    productivity_score DECIMAL(4,2),
    digital_addiction_score DECIMAL(5,2),
    brain_rot_index DECIMAL(5,2),
    academic_risk_score DECIMAL(5,2),
    usage_level VARCHAR(20),
    FOREIGN KEY (country_id) REFERENCES countries(country_id)
);

-- ----------------------------------------------------------------------------
-- 3. ANALYTICAL SQL QUERIES (8 Core Requirements)
-- ----------------------------------------------------------------------------

-- Query 1: Basic SELECT & FROM
-- Purpose: Retrieve initial sample records showing demographic and daily activity metrics.
SELECT student_id, age, gender, device_access, social_media_hours, study_hours_per_week, productivity_score
FROM students
LIMIT 5;

-- Query 2: WHERE Filtering
-- Purpose: Filter for high daily screen time users who report spending more than 6 hours per day.
-- Result: 14,103 students (2.82% of dataset)
SELECT student_id, age, gender, social_media_hours, study_hours_per_week, productivity_score
FROM students
WHERE social_media_hours > 6.0
LIMIT 5;

-- Query 3: Multiple Conditions with WHERE (AND / OR)
-- Purpose: Identify high-risk cohort: screen time > 5 hrs, addiction score > 20, and low motivation (< 4.0).
-- Result: 10,224 students (2.04% of total sample)
SELECT COUNT(*) AS high_risk_distraction_cohort
FROM students
WHERE social_media_hours > 5.0 
  AND digital_addiction_score > 20.0 
  AND academic_motivation < 4.0;

-- Query 4: ORDER BY Sorting
-- Purpose: Rank students by highest weekly study hours to inspect high academic diligence.
SELECT student_id, study_hours_per_week, social_media_hours, productivity_score, usage_level
FROM students
ORDER BY study_hours_per_week DESC, social_media_hours ASC
LIMIT 5;

-- Query 5: Aggregation (COUNT, AVG, MIN, MAX)
-- Purpose: Calculate global summary statistics across all 500,000 respondents.
-- Result: Total=500,000 | Avg SM=3.36h | Avg Study=22.98h | Avg Prod=8.83 | Min SM=0.00 | Max SM=9.00
SELECT 
    COUNT(*) AS total_students,
    ROUND(AVG(social_media_hours), 2) AS avg_screen_time_hours,
    ROUND(AVG(study_hours_per_week), 2) AS avg_weekly_study_hours,
    ROUND(AVG(productivity_score), 2) AS avg_productivity_score,
    MIN(social_media_hours) AS min_screen_time,
    MAX(social_media_hours) AS max_screen_time
FROM students;

-- Query 6: GROUP BY Comparison (Source for Visualization 2)
-- Purpose: Aggregate student counts, study hours, and productivity across usage tiers.
-- Result:
--   Low (<=3.0h):       N=204,663 (40.9%) | Avg Study=23.77h | Avg Prod=9.31 | Avg Addict=9.53
--   Moderate (3.0-5.0h): N=241,219 (48.2%) | Avg Study=22.71h | Avg Prod=8.67 | Avg Addict=15.91
--   High (>5.0h):       N=54,118  (10.8%) | Avg Study=21.23h | Avg Prod=7.72 | Avg Addict=23.07
SELECT 
    usage_level,
    COUNT(*) AS student_count,
    ROUND(AVG(study_hours_per_week), 2) AS avg_study_hours,
    ROUND(AVG(productivity_score), 2) AS avg_productivity,
    ROUND(AVG(digital_addiction_score), 2) AS avg_addiction
FROM students
GROUP BY usage_level
ORDER BY student_count DESC;

-- Query 7: HAVING Clause
-- Purpose: Identify nations where average social media screen time exceeds 3.40 hours with cohort >= 9,500.
-- Result: 24 nations qualify, led by USA (4.29h), UK (4.26h), Japan (4.25h), Germany (4.23h), Switzerland (4.21h).
SELECT 
    c.country_name,
    COUNT(s.student_id) AS student_count,
    ROUND(AVG(s.social_media_hours), 2) AS avg_screen_time,
    ROUND(AVG(s.study_hours_per_week), 2) AS avg_study_time
FROM students s
JOIN countries c ON s.country_id = c.country_id
GROUP BY c.country_name
HAVING COUNT(s.student_id) >= 9500 AND AVG(s.social_media_hours) > 3.40
ORDER BY avg_screen_time DESC;

-- Query 8: Relational JOIN
-- Purpose: Inner join students and countries to compare digital habits across national development tiers.
-- Result:
--   Developing:     N=210,346 (42.1%) | Avg SM=3.33h | Avg Study=23.00h | Avg Prod=8.85
--   Developed:      N=149,548 (29.9%) | Avg SM=4.03h | Avg Study=23.28h | Avg Prod=8.59
--   Underdeveloped: N=140,106 (28.0%) | Avg SM=2.69h | Avg Study=22.64h | Avg Prod=9.05
SELECT 
    c.development_level,
    COUNT(s.student_id) AS total_students,
    ROUND(AVG(s.social_media_hours), 2) AS avg_social_media_hours,
    ROUND(AVG(s.study_hours_per_week), 2) AS avg_weekly_study_hours,
    ROUND(AVG(s.productivity_score), 2) AS avg_productivity_score
FROM students s
INNER JOIN countries c ON s.country_id = c.country_id
GROUP BY c.development_level
ORDER BY total_students DESC;
