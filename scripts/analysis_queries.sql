-- ============================================================================
-- Assignment 4: SQL Relational Data Analysis
-- Course Project: Student Social Media Usage & Academic Performance
-- Database Schema & Analytical Queries (N = 405)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. TABLE CREATION: Social Media Platforms (Lookup / Dimension Table)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS social_media_platforms;

CREATE TABLE social_media_platforms (
    platform_id INT PRIMARY KEY,
    platform_name VARCHAR(50) NOT NULL,
    category_type VARCHAR(50),
    primary_format VARCHAR(50)
);

-- Seed Data: Platform Information
INSERT INTO social_media_platforms (platform_id, platform_name, category_type, primary_format) VALUES
(1, 'Facebook', 'Social Networking', 'Feed, Groups & Posts'),
(2, 'Instagram', 'Visual & Media Sharing', 'Photos, Stories & Reels'),
(3, 'YouTube', 'Video Streaming', 'Long & Short-Form Video'),
(4, 'Other', 'Alternative Platforms', 'Multi-Network / Messaging'),
(5, 'TikTok', 'Short-Form Video', 'Algorithmic Video Feed');

-- ----------------------------------------------------------------------------
-- 2. TABLE CREATION: Students (Main Entity / Fact Table)
-- ----------------------------------------------------------------------------
CREATE TABLE students (
    student_id INT PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    screen_time_hours DECIMAL(4,2),
    study_time_hours DECIMAL(4,2),
    platform_id INT,
    academic_distraction_scale INT,
    gpa DECIMAL(3,2),
    usage_level VARCHAR(15),
    FOREIGN KEY (platform_id) REFERENCES social_media_platforms(platform_id)
);

-- ----------------------------------------------------------------------------
-- 3. ANALYTICAL SQL QUERIES (8 Core Requirements)
-- ----------------------------------------------------------------------------

-- Query 1: Basic SELECT & FROM
-- Purpose: Retrieve initial sample records showing demographic and daily activity metrics.
SELECT student_id, gender, age, screen_time_hours, study_time_hours
FROM students
LIMIT 5;

-- Query 2: WHERE Filtering
-- Purpose: Filter for high daily screen time users who report spending more than 7 hours per day.
SELECT student_id, age, gender, screen_time_hours, gpa
FROM students
WHERE screen_time_hours > 7.0
LIMIT 5;

-- Query 3: Multiple Conditions with WHERE (AND / OR)
-- Purpose: Identify students with high screen time (>= 5 hrs) who also experience elevated distraction (scale >= 4).
SELECT COUNT(*) AS high_screen_and_distraction_count
FROM students
WHERE screen_time_hours >= 5.0 AND academic_distraction_scale >= 4;

-- Query 4: ORDER BY Sorting
-- Purpose: Rank students by highest daily screen time to inspect top screen time consumers.
SELECT student_id, screen_time_hours, study_time_hours, gpa, usage_level
FROM students
ORDER BY screen_time_hours DESC, study_time_hours ASC
LIMIT 5;

-- Query 5: Aggregation (COUNT, AVG, MIN, MAX)
-- Purpose: Calculate overall summary statistics for daily screen time and study duration across all 405 participants.
SELECT 
    COUNT(*) AS total_students,
    ROUND(AVG(screen_time_hours), 2) AS avg_screen_time,
    ROUND(AVG(study_time_hours), 2) AS avg_study_time,
    MIN(screen_time_hours) AS min_screen_time,
    MAX(screen_time_hours) AS max_screen_time
FROM students;

-- Query 6: GROUP BY Comparison (Source for Visualization 2)
-- Purpose: Aggregate student counts and average study hours grouped by usage tier (Low, Moderate, High).
SELECT 
    usage_level,
    COUNT(*) AS student_count,
    ROUND(AVG(study_time_hours), 2) AS avg_study_hours
FROM students
GROUP BY usage_level
ORDER BY student_count DESC;

-- Query 7: HAVING Clause
-- Purpose: Filter grouped age brackets to inspect only groups with at least 15 respondents.
SELECT 
    age,
    COUNT(*) AS student_count,
    ROUND(AVG(screen_time_hours), 2) AS avg_screen_time
FROM students
GROUP BY age
HAVING COUNT(*) >= 15
ORDER BY student_count DESC;

-- Query 8: JOIN Between Students and Social Media Platforms (Source for Visualization 1)
-- Purpose: Combine student records with platform metadata to aggregate usage by platform name and category.
SELECT 
    p.platform_name,
    p.category_type,
    COUNT(s.student_id) AS student_count,
    ROUND(AVG(s.screen_time_hours), 2) AS avg_screen_hours
FROM students s
JOIN social_media_platforms p ON s.platform_id = p.platform_id
GROUP BY p.platform_id, p.platform_name, p.category_type
ORDER BY student_count DESC;
