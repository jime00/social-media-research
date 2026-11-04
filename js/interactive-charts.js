/**
 * ==============================================================================
 * SOCIAL MEDIA USAGE RESEARCH - INTERACTIVE VISUALIZATIONS (PLOTLY.JS)
 * Description: Client-side interactive data visualizations using Plotly.js.
 *              Renders 3 responsive charts from empirical 500,000-student dataset:
 *                1. Interactive Scatter Plot (Screen Time vs Attention Span)
 *                2. Interactive Bar Chart (Cross-Group Academic & Digital Comparisons)
 *                3. Interactive Box Plot (Sleep & Study Hour Distributions)
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
    // Check if Plotly and data payload are available
    if (typeof Plotly === "undefined") {
        console.warn("Plotly library not loaded. Retrying in 500ms...");
        setTimeout(initInteractiveCharts, 500);
        return;
    }
    initInteractiveCharts();
});

function initInteractiveCharts() {
    const data = window.RESEARCH_INTERACTIVE_DATA;
    if (!data) {
        console.error("RESEARCH_INTERACTIVE_DATA payload not found.");
        return;
    }

    console.log("Initializing Plotly interactive visualizations with empirical data...");

    // Color definitions aligned with design system
    const colorPalette = {
        primary: "#bc5689",
        navyDark: "#1b2d42",
        navyMedium: "#2a4365",
        accentTeal: "#0d9488",
        accentAmber: "#d97706",
        accentRose: "#e11d48",
        eduColors: {
            "School": "#bc5689",
            "Diploma": "#2a4365",
            "Graduate": "#0d9488",
            "Postgraduate": "#1b2d42",
            "Dropout": "#d97706",
            "PhD": "#8b5cf6"
        },
        tierColors: {
            "Low (<= 3 hrs)": "#059669",
            "Moderate (3-5 hrs)": "#d97706",
            "High (> 5 hrs)": "#dc2626"
        }
    };

    // Shared layout configuration for crisp academic aesthetic
    const sharedConfig = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    /* --------------------------------------------------------------------------
       1. INTERACTIVE SCATTER PLOT: Social Media Hours vs. Attention Span
       Features: Hover tooltips, Dropdown filtering by Education Level, Zoom/Pan
       -------------------------------------------------------------------------- */
    const scatterContainer = document.getElementById("interactive-scatter");
    if (scatterContainer) {
        const samples = data.scatter_sample;
        const eduLevels = data.education_levels;

        // Create separate traces per education level for dynamic filtering
        const scatterTraces = eduLevels.map(edu => {
            const eduSamples = samples.filter(s => s.edu === edu);
            return {
                x: eduSamples.map(s => s.sm_hours),
                y: eduSamples.map(s => s.attention),
                mode: 'markers',
                name: edu,
                marker: {
                    size: 7,
                    color: colorPalette.eduColors[edu] || colorPalette.primary,
                    opacity: 0.65,
                    line: { width: 1, color: '#ffffff' }
                },
                customdata: eduSamples.map(s => [s.id, s.country, s.brain_rot, s.study_hours, s.prod_score]),
                hovertemplate:
                    '<b>Student #%{customdata[0]}</b> (%{data.name}, %{customdata[1]})<br>' +
                    '📱 Screen Time: <b>%{x:.2f} hrs/day</b><br>' +
                    '🧠 Attention Span: <b>%{y:.1f} mins</b><br>' +
                    '⚡ Brain Rot Index: %{customdata[2]}<br>' +
                    '📚 Weekly Study: %{customdata[3]} hrs<br>' +
                    '🎯 Productivity: %{customdata[4]}/10' +
                    '<extra></extra>'
            };
        });

        // Dropdown menu options for filtering cohorts
        const updatemenus = [{
            buttons: [
                {
                    method: 'restyle',
                    args: ['visible', [true, true, true, true, true, true]],
                    label: 'All Education Levels (Full Sample)'
                },
                ...eduLevels.map((edu, idx) => {
                    const vis = [false, false, false, false, false, false];
                    vis[idx] = true;
                    return {
                        method: 'restyle',
                        args: ['visible', vis],
                        label: `Show: ${edu} Only`
                    };
                })
            ],
            direction: 'down',
            pad: { r: 10, t: 10 },
            showactive: true,
            x: 0.0,
            xanchor: 'left',
            y: 1.16,
            yanchor: 'top',
            bgcolor: '#ffffff',
            bordercolor: '#cbd5e1',
            font: { size: 12, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' }
        }];

        const scatterLayout = {
            title: {
                text: '<b>Bivariate Correlation: Social Media Hours vs. Sustained Attention Span</b>',
                font: { size: 16, color: colorPalette.navyDark, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
                x: 0.05,
                y: 0.98
            },
            xaxis: {
                title: { text: 'Daily Social Media Usage (Hours)', font: { size: 13, color: colorPalette.navyDark } },
                range: [0, 9.5],
                gridcolor: '#e2e8f0',
                zerolinecolor: '#cbd5e1'
            },
            yaxis: {
                title: { text: 'Attention Span (Minutes)', font: { size: 13, color: colorPalette.navyDark } },
                range: [20, 65],
                gridcolor: '#e2e8f0',
                zerolinecolor: '#cbd5e1'
            },
            plot_bgcolor: '#f8fafc',
            paper_bgcolor: '#ffffff',
            hovermode: 'closest',
            updatemenus: updatemenus,
            legend: {
                orientation: 'h',
                x: 0.0,
                y: -0.22,
                font: { size: 11, color: '#57606a' }
            },
            margin: { t: 90, r: 25, b: 70, l: 60 }
        };

        Plotly.newPlot(scatterContainer, scatterTraces, scatterLayout, sharedConfig);
    }

    /* --------------------------------------------------------------------------
       2. INTERACTIVE BAR CHART: Comparing Student Groups Across Key Metrics
       Features: Metric switcher dropdown, Hover data cards, Group comparison
       -------------------------------------------------------------------------- */
    const barContainer = document.getElementById("interactive-barchart");
    if (barContainer) {
        const eduAgg = data.education_aggregates;
        const eduKeys = data.education_levels;

        // Metric configurations available in the dropdown
        const metricConfigs = {
            sm_hours: {
                label: 'Average Daily Social Media Hours',
                yTitle: 'Daily Screen Time (Hours)',
                color: '#bc5689',
                extract: edu => eduAgg[edu].sm_hours.mean,
                format: v => `${v.toFixed(2)} hrs/day`
            },
            attention: {
                label: 'Average Attention Span (Minutes)',
                yTitle: 'Attention Span (Minutes)',
                color: '#2a4365',
                extract: edu => eduAgg[edu].attention.mean,
                format: v => `${v.toFixed(1)} mins`
            },
            study_hours: {
                label: 'Average Weekly Study Hours',
                yTitle: 'Study Duration (Hours/Week)',
                color: '#0d9488',
                extract: edu => eduAgg[edu].study_hours.mean,
                format: v => `${v.toFixed(2)} hrs/week`
            },
            sleep_hours: {
                label: 'Average Sleep Duration (Hours/Night)',
                yTitle: 'Sleep Duration (Hours)',
                color: '#1b2d42',
                extract: edu => eduAgg[edu].sleep_hours.mean,
                format: v => `${v.toFixed(2)} hrs/night`
            },
            brain_rot: {
                label: 'Average Brain Rot / Cognitive Fatigue Index',
                yTitle: 'Fatigue Index Score (0–100)',
                color: '#d97706',
                extract: edu => eduAgg[edu].brain_rot.mean,
                format: v => `${v.toFixed(1)} pts`
            },
            risk_rate: {
                label: 'Academic Risk Rate (% with Risk Score > 0)',
                yTitle: 'Cohort Academic Risk Rate (%)',
                color: '#dc2626',
                extract: edu => eduAgg[edu].academic_risk_rate_pct,
                format: v => `${v.toFixed(2)}%`
            }
        };

        const defaultMetric = 'sm_hours';
        const defaultCfg = metricConfigs[defaultMetric];

        const barTrace = {
            x: eduKeys,
            y: eduKeys.map(k => defaultCfg.extract(k)),
            type: 'bar',
            marker: {
                color: eduKeys.map(k => colorPalette.eduColors[k] || defaultCfg.color),
                line: { width: 1.5, color: '#1b2d42' },
                opacity: 0.9
            },
            text: eduKeys.map(k => defaultCfg.format(defaultCfg.extract(k))),
            textposition: 'outside',
            customdata: eduKeys.map(k => eduAgg[k].count.toLocaleString()),
            hovertemplate:
                '<b>%{x} Cohort</b> (N = %{customdata})<br>' +
                'Value: <b>%{text}</b>' +
                '<extra></extra>'
        };

        // Dropdown menu to change metrics
        const metricButtons = Object.entries(metricConfigs).map(([key, cfg]) => {
            const yVals = eduKeys.map(k => cfg.extract(k));
            const textVals = eduKeys.map(k => cfg.format(cfg.extract(k)));
            return {
                method: 'update',
                args: [
                    {
                        y: [yVals],
                        text: [textVals]
                    },
                    {
                        'yaxis.title.text': cfg.yTitle,
                        'title.text': `<b>Cross-Group Analysis: ${cfg.label} by Education Level</b>`
                    }
                ],
                label: cfg.label
            };
        });

        const barLayout = {
            title: {
                text: `<b>Cross-Group Analysis: ${defaultCfg.label} by Education Level</b>`,
                font: { size: 16, color: colorPalette.navyDark, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
                x: 0.05,
                y: 0.98
            },
            xaxis: {
                title: { text: 'Education Level', font: { size: 13, color: colorPalette.navyDark } },
                tickfont: { size: 12 }
            },
            yaxis: {
                title: { text: defaultCfg.yTitle, font: { size: 13, color: colorPalette.navyDark } },
                gridcolor: '#e2e8f0'
            },
            plot_bgcolor: '#f8fafc',
            paper_bgcolor: '#ffffff',
            updatemenus: [{
                buttons: metricButtons,
                direction: 'down',
                pad: { r: 10, t: 10 },
                showactive: true,
                x: 0.0,
                xanchor: 'left',
                y: 1.18,
                yanchor: 'top',
                bgcolor: '#ffffff',
                bordercolor: '#cbd5e1',
                font: { size: 12, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' }
            }],
            margin: { t: 95, r: 25, b: 60, l: 60 }
        };

        Plotly.newPlot(barContainer, [barTrace], barLayout, sharedConfig);
    }

    /* --------------------------------------------------------------------------
       3. INTERACTIVE BOX PLOT: Sleep Hours & Study Hours Across Student Groups
       Features: Variable switcher (Sleep vs Study), Full 5-number summary on hover
       -------------------------------------------------------------------------- */
    const boxContainer = document.getElementById("interactive-boxplot");
    if (boxContainer) {
        const eduAgg = data.education_aggregates;
        const usageAgg = data.usage_aggregates;
        const eduKeys = data.education_levels;
        const usageKeys = data.usage_tiers;

        // Build box trace function for pre-aggregated five-number statistics
        function makeBoxTraces(keys, aggObj, statKey, traceColor) {
            return keys.map((key, i) => {
                const box = aggObj[key][statKey];
                return {
                    type: 'box',
                    name: key,
                    q1: [box.q1],
                    median: [box.median],
                    q3: [box.q3],
                    lowerfence: [box.min],
                    upperfence: [box.max],
                    mean: [box.mean],
                    boxpoints: false,
                    marker: { color: traceColor || colorPalette.primary },
                    line: { width: 1.8 },
                    fillcolor: 'rgba(188, 86, 137, 0.25)',
                    hovertemplate:
                        '<b>%{x}</b><br>' +
                        'Max: %{upperfence:.2f}<br>' +
                        'Q3: %{q3:.2f}<br>' +
                        'Median: <b>%{median:.2f}</b><br>' +
                        'Mean: %{mean:.2f}<br>' +
                        'Q1: %{q1:.2f}<br>' +
                        'Min: %{lowerfence:.2f}' +
                        '<extra></extra>'
                };
            });
        }

        // Trace set 1: Sleep Hours by Education Level (default)
        const tracesSleepEdu = makeBoxTraces(eduKeys, eduAgg, 'sleep_hours', '#1b2d42');
        // Trace set 2: Study Hours by Education Level
        const tracesStudyEdu = makeBoxTraces(eduKeys, eduAgg, 'study_hours', '#0d9488');
        // Trace set 3: Sleep Hours by Usage Tier
        const tracesSleepUsage = makeBoxTraces(usageKeys, usageAgg, 'sleep_hours', '#bc5689');
        // Trace set 4: Study Hours by Usage Tier
        const tracesStudyUsage = makeBoxTraces(usageKeys, usageAgg, 'study_hours', '#d97706');

        // Initial setup: render Sleep by Education
        const initialTraces = tracesSleepEdu;

        const boxButtons = [
            {
                method: 'restyle',
                args: [
                    {
                        name: eduKeys,
                        q1: eduKeys.map(k => [eduAgg[k].sleep_hours.q1]),
                        median: eduKeys.map(k => [eduAgg[k].sleep_hours.median]),
                        q3: eduKeys.map(k => [eduAgg[k].sleep_hours.q3]),
                        lowerfence: eduKeys.map(k => [eduAgg[k].sleep_hours.min]),
                        upperfence: eduKeys.map(k => [eduAgg[k].sleep_hours.max]),
                        mean: eduKeys.map(k => [eduAgg[k].sleep_hours.mean]),
                        'marker.color': '#1b2d42'
                    }
                ],
                label: 'Sleep Hours by Education Level'
            },
            {
                method: 'restyle',
                args: [
                    {
                        name: eduKeys,
                        q1: eduKeys.map(k => [eduAgg[k].study_hours.q1]),
                        median: eduKeys.map(k => [eduAgg[k].study_hours.median]),
                        q3: eduKeys.map(k => [eduAgg[k].study_hours.q3]),
                        lowerfence: eduKeys.map(k => [eduAgg[k].study_hours.min]),
                        upperfence: eduKeys.map(k => [eduAgg[k].study_hours.max]),
                        mean: eduKeys.map(k => [eduAgg[k].study_hours.mean]),
                        'marker.color': '#0d9488'
                    }
                ],
                label: 'Weekly Study Hours by Education Level'
            }
        ];

        const boxLayout = {
            title: {
                text: '<b>Five-Number Summary Distribution: Sleep & Study Hours across Student Groups</b>',
                font: { size: 16, color: colorPalette.navyDark, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
                x: 0.05,
                y: 0.98
            },
            yaxis: {
                title: { text: 'Duration (Hours)', font: { size: 13, color: colorPalette.navyDark } },
                gridcolor: '#e2e8f0'
            },
            xaxis: {
                title: { text: 'Student Cohort', font: { size: 13, color: colorPalette.navyDark } }
            },
            plot_bgcolor: '#f8fafc',
            paper_bgcolor: '#ffffff',
            showlegend: false,
            updatemenus: [{
                buttons: boxButtons,
                direction: 'down',
                pad: { r: 10, t: 10 },
                showactive: true,
                x: 0.0,
                xanchor: 'left',
                y: 1.18,
                yanchor: 'top',
                bgcolor: '#ffffff',
                bordercolor: '#cbd5e1',
                font: { size: 12, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' }
            }],
            margin: { t: 95, r: 25, b: 60, l: 60 }
        };

        Plotly.newPlot(boxContainer, initialTraces, boxLayout, sharedConfig);
    }
}
