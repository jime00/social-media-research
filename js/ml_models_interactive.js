/**
 * ==============================================================================
 * SOCIAL MEDIA USAGE RESEARCH - INTERACTIVE MACHINE LEARNING VISUALIZATIONS
 * Description: Client-side interactive visualizations using Plotly.js for:
 *              1. Linear Regression (Main Model): Actual vs. Predicted, R² = 0.0847, 80/20 split
 *              2. K-Means Clustering (Secondary Model): 3 clusters, centroids, and averages
 * Source: Actual student records from data/global_student_digital_behavior_dataset.csv
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
    // Retry initialization if Plotly is still loading
    if (typeof Plotly === "undefined") {
        console.warn("Plotly library not yet available. Retrying in 400ms...");
        setTimeout(initMLInteractiveCharts, 400);
        return;
    }
    initMLInteractiveCharts();
});

function initMLInteractiveCharts() {
    const mlData = window.RESEARCH_ML_MODELS_DATA;
    if (!mlData) {
        console.error("RESEARCH_ML_MODELS_DATA payload not found.");
        return;
    }

    console.log("Initializing interactive Machine Learning visualizations...");

    const colorPalette = {
        primary: "#bc5689",
        navyDark: "#1b2d42",
        navyMedium: "#2a4365",
        cluster0: "#059669",  // Focused Academic (Emerald)
        cluster1: "#2a4365",  // Balanced / Moderate (Navy)
        cluster2: "#bc5689",  // High Digital Fatigue (Plum)
        centroid: "#d97706"   // Centroid Amber
    };

    const sharedConfig = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    /* --------------------------------------------------------------------------
       1. INTERACTIVE LINEAR REGRESSION VISUALIZATION (MAIN MODEL)
       Features: Observed actual points, fitted regression line, actual vs. predicted,
                 R² = 0.0847, 80/20 train/test split, interactive feature switching
       -------------------------------------------------------------------------- */
    const regContainer = document.getElementById("interactive-linear-regression");
    if (regContainer) {
        const lr = mlData.linear_regression;
        const pts = lr.test_sample;

        // Trace 1: Actual Observed Test Points
        const traceObserved = {
            x: pts.map(p => p.predicted_risk),
            y: pts.map(p => p.actual_risk),
            mode: 'markers',
            name: 'Observed Test Records (N=1,000)',
            marker: {
                size: 7,
                color: colorPalette.primary,
                opacity: 0.6,
                line: { width: 1, color: '#9d174d' }
            },
            customdata: pts.map(p => [
                p.id, p.country, p.residual,
                p.sm_hours, p.brain_rot, p.attention,
                p.stress, p.sleep, p.predicted_risk
            ]),
            hovertemplate:
                '<b>Student #%{customdata[0]}</b> (%{customdata[1]})<br>' +
                '🎯 Actual Risk Score: <b>%{y:.2f}</b><br>' +
                '📈 Predicted Risk: <b>%{x:.2f}</b><br>' +
                '⚠️ Residual Error: <b>%{customdata[2]:+.2f}</b><br>' +
                '📱 Screen Time: %{customdata[3]} hrs/day<br>' +
                '⚡ Brain Rot Index: %{customdata[4]}<br>' +
                '🧠 Attention Span: %{customdata[5]} mins<br>' +
                '🔥 Stress Level: %{customdata[6]} / 10<br>' +
                '🌙 Sleep: %{customdata[7]} hrs/night' +
                '<extra></extra>'
        };

        // Trace 2: Fitted 45-degree Reference Line (y = x)
        const traceFitLine = {
            x: [0.0, 1.8],
            y: [0.0, 1.8],
            mode: 'lines',
            name: 'Ideal Fit Line (y = ŷ)',
            line: {
                color: '#64748b',
                width: 2.5,
                dash: 'dash'
            },
            hoverinfo: 'none'
        };

        // Dropdown menu to switch regression views
        const regButtons = [
            {
                method: 'update',
                args: [
                    {
                        x: [pts.map(p => p.predicted_risk), [0.0, 1.8]],
                        y: [pts.map(p => p.actual_risk), [0.0, 1.8]],
                        'marker.color': [colorPalette.primary, null]
                    },
                    {
                        'xaxis.title.text': 'Predicted Academic Risk Score (OLS Multi-Feature Output)',
                        'yaxis.title.text': 'Actual Academic Risk Score (Ground Truth)',
                        'title.text': '<b>Linear Regression: Actual vs. Predicted Academic Risk Score (R² = 0.0847)</b>'
                    }
                ],
                label: 'View 1: Actual vs. Predicted Risk Scores (Default)'
            },
            {
                method: 'update',
                args: [
                    {
                        x: [pts.map(p => p.sm_hours), [0.0, 9.0]],
                        y: [pts.map(p => p.actual_risk), [lr.fitted_lines.screen_time_vs_risk[0].y, lr.fitted_lines.screen_time_vs_risk[1].y]],
                        'marker.color': ['#2a4365', null]
                    },
                    {
                        'xaxis.title.text': 'Daily Social Media Hours (hrs/day)',
                        'yaxis.title.text': 'Academic Risk Score',
                        'title.text': '<b>Linear Regression: Daily Screen Time vs. Academic Risk Score</b>'
                    }
                ],
                label: 'View 2: Screen Time vs. Academic Risk'
            },
            {
                method: 'update',
                args: [
                    {
                        x: [pts.map(p => p.brain_rot), [0.0, 48.0]],
                        y: [pts.map(p => p.actual_risk), [lr.fitted_lines.brain_rot_vs_risk[0].y, lr.fitted_lines.brain_rot_vs_risk[1].y]],
                        'marker.color': ['#d97706', null]
                    },
                    {
                        'xaxis.title.text': 'Brain Rot Index Score',
                        'yaxis.title.text': 'Academic Risk Score',
                        'title.text': '<b>Linear Regression: Brain Rot Index vs. Academic Risk Score</b>'
                    }
                ],
                label: 'View 3: Brain Rot Index vs. Academic Risk'
            },
            {
                method: 'update',
                args: [
                    {
                        x: [pts.map(p => p.attention), [20.0, 65.0]],
                        y: [pts.map(p => p.actual_risk), [1.2, 0.05]],
                        'marker.color': ['#0d9488', null]
                    },
                    {
                        'xaxis.title.text': 'Sustained Attention Span (minutes)',
                        'yaxis.title.text': 'Academic Risk Score',
                        'title.text': '<b>Linear Regression: Attention Span vs. Academic Risk Score</b>'
                    }
                ],
                label: 'View 4: Attention Span vs. Academic Risk'
            }
        ];

        const regLayout = {
            title: {
                text: '<b>Linear Regression: Actual vs. Predicted Academic Risk Score (R² = 0.0847)</b>',
                font: { size: 16, color: colorPalette.navyDark, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
                x: 0.05,
                y: 0.98
            },
            xaxis: {
                title: { text: 'Predicted Academic Risk Score (OLS Multi-Feature Output)', font: { size: 13, color: colorPalette.navyDark } },
                gridcolor: '#e2e8f0',
                zerolinecolor: '#cbd5e1'
            },
            yaxis: {
                title: { text: 'Actual Academic Risk Score (Ground Truth)', font: { size: 13, color: colorPalette.navyDark } },
                gridcolor: '#e2e8f0',
                zerolinecolor: '#cbd5e1'
            },
            plot_bgcolor: '#f8fafc',
            paper_bgcolor: '#ffffff',
            hovermode: 'closest',
            updatemenus: [{
                buttons: regButtons,
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
            legend: {
                orientation: 'h',
                x: 0.0,
                y: -0.22,
                font: { size: 11, color: '#57606a' }
            },
            annotations: [
                {
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.98,
                    y: 0.94,
                    xanchor: 'right',
                    yanchor: 'top',
                    text: '<b>Holdout Evaluation:</b><br>• R² Score: <b>0.0847</b><br>• Split: <b>80% Train / 20% Test</b><br>• Features: <b>5 Variables</b>',
                    showarrow: false,
                    font: { size: 11, color: colorPalette.navyDark },
                    bgcolor: '#ffffff',
                    bordercolor: '#cbd5e1',
                    borderwidth: 1,
                    borderpad: 6
                }
            ],
            margin: { t: 95, r: 25, b: 70, l: 60 }
        };

        Plotly.newPlot(regContainer, [traceObserved, traceFitLine], regLayout, sharedConfig);
    }

    /* --------------------------------------------------------------------------
       2. INTERACTIVE K-MEANS CLUSTERING VISUALIZATION (SECONDARY MODEL)
       Features: 3 distinct cluster colors, individual student points, cluster centroids,
                 cluster averages, interactive feature-axis switching
       -------------------------------------------------------------------------- */
    const kmContainer = document.getElementById("interactive-kmeans-clustering");
    if (kmContainer) {
        const km = mlData.kmeans_clustering;
        const students = km.student_sample;
        const summaries = km.cluster_summaries;

        // Separate traces per cluster
        const clusterTraces = summaries.map(c => {
            const cStudents = students.filter(s => s.cluster_id === c.cluster_id);
            return {
                x: cStudents.map(s => s.sm_hours),
                y: cStudents.map(s => s.study_hours),
                mode: 'markers',
                name: `${c.name} (${c.percentage}%)`,
                marker: {
                    size: 7,
                    color: c.color,
                    opacity: 0.65,
                    line: { width: 1, color: '#ffffff' }
                },
                customdata: cStudents.map(s => [
                    s.id, s.country, s.sm_hours, s.brain_rot, s.attention, s.study_hours, c.name
                ]),
                hovertemplate:
                    '<b>%{customdata[6]}</b><br>' +
                    'Student #%{customdata[0]} (%{customdata[1]})<br>' +
                    '📱 Screen Time: <b>%{x:.2f} hrs/day</b><br>' +
                    '📚 Weekly Study: <b>%{y:.2f} hrs/week</b><br>' +
                    '⚡ Brain Rot Index: %{customdata[3]}<br>' +
                    '🧠 Attention Span: %{customdata[4]} mins' +
                    '<extra></extra>'
            };
        });

        // Trace 4: Cluster Centroids
        const centroidTrace = {
            x: summaries.map(c => c.avg_sm_hours),
            y: summaries.map(c => c.avg_study_hours),
            mode: 'markers+text',
            name: 'Cluster Centroids (Centers)',
            text: summaries.map(c => `<b>Center ${c.cluster_id + 1}</b>`),
            textposition: 'top center',
            textfont: { size: 11, color: colorPalette.navyDark },
            marker: {
                size: 15,
                color: '#f59e0b',
                symbol: 'diamond-wide-dot',
                line: { width: 2.5, color: '#1b2d42' }
            },
            customdata: summaries.map(c => [
                c.name, c.avg_sm_hours, c.avg_brain_rot, c.avg_attention, c.avg_study_hours, c.count
            ]),
            hovertemplate:
                '<b>📍 Cluster Center: %{customdata[0]}</b><br>' +
                '👥 Students in Cluster: %{customdata[5]:,}<br>' +
                '📱 Avg Screen Time: <b>%{x:.2f} hrs/day</b><br>' +
                '📚 Avg Weekly Study: <b>%{y:.2f} hrs/week</b><br>' +
                '⚡ Avg Brain Rot Index: <b>%{customdata[2]}</b><br>' +
                '🧠 Avg Attention Span: <b>%{customdata[3]} mins</b>' +
                '<extra></extra>'
        };

        const allKmTraces = [...clusterTraces, centroidTrace];

        // Dropdown menu to switch feature axes
        const kmButtons = [
            {
                method: 'update',
                args: [
                    {
                        x: [
                            ...summaries.map(c => students.filter(s => s.cluster_id === c.cluster_id).map(s => s.sm_hours)),
                            summaries.map(c => c.avg_sm_hours)
                        ],
                        y: [
                            ...summaries.map(c => students.filter(s => s.cluster_id === c.cluster_id).map(s => s.study_hours)),
                            summaries.map(c => c.avg_study_hours)
                        ]
                    },
                    {
                        'xaxis.title.text': 'Daily Social Media Hours (hrs/day)',
                        'yaxis.title.text': 'Weekly Study Hours (hrs/week)',
                        'title.text': '<b>K-Means Clustering: Screen Time vs. Weekly Study Hours (k = 3)</b>'
                    }
                ],
                label: 'Axis 1: Screen Time vs. Study Hours (Default)'
            },
            {
                method: 'update',
                args: [
                    {
                        x: [
                            ...summaries.map(c => students.filter(s => s.cluster_id === c.cluster_id).map(s => s.brain_rot)),
                            summaries.map(c => c.avg_brain_rot)
                        ],
                        y: [
                            ...summaries.map(c => students.filter(s => s.cluster_id === c.cluster_id).map(s => s.attention)),
                            summaries.map(c => c.avg_attention)
                        ]
                    },
                    {
                        'xaxis.title.text': 'Brain Rot Index Score',
                        'yaxis.title.text': 'Sustained Attention Span (minutes)',
                        'title.text': '<b>K-Means Clustering: Brain Rot Index vs. Attention Span (k = 3)</b>'
                    }
                ],
                label: 'Axis 2: Brain Rot vs. Attention Span'
            },
            {
                method: 'update',
                args: [
                    {
                        x: [
                            ...summaries.map(c => students.filter(s => s.cluster_id === c.cluster_id).map(s => s.sm_hours)),
                            summaries.map(c => c.avg_sm_hours)
                        ],
                        y: [
                            ...summaries.map(c => students.filter(s => s.cluster_id === c.cluster_id).map(s => s.brain_rot)),
                            summaries.map(c => c.avg_brain_rot)
                        ]
                    },
                    {
                        'xaxis.title.text': 'Daily Social Media Hours (hrs/day)',
                        'yaxis.title.text': 'Brain Rot Index Score',
                        'title.text': '<b>K-Means Clustering: Screen Time vs. Brain Rot Index (k = 3)</b>'
                    }
                ],
                label: 'Axis 3: Screen Time vs. Brain Rot'
            },
            {
                method: 'update',
                args: [
                    {
                        x: [
                            ...summaries.map(c => students.filter(s => s.cluster_id === c.cluster_id).map(s => s.attention)),
                            summaries.map(c => c.avg_attention)
                        ],
                        y: [
                            ...summaries.map(c => students.filter(s => s.cluster_id === c.cluster_id).map(s => s.study_hours)),
                            summaries.map(c => c.avg_study_hours)
                        ]
                    },
                    {
                        'xaxis.title.text': 'Sustained Attention Span (minutes)',
                        'yaxis.title.text': 'Weekly Study Hours (hrs/week)',
                        'title.text': '<b>K-Means Clustering: Attention Span vs. Weekly Study Hours (k = 3)</b>'
                    }
                ],
                label: 'Axis 4: Attention Span vs. Study Hours'
            }
        ];

        const kmLayout = {
            title: {
                text: '<b>K-Means Clustering: Screen Time vs. Weekly Study Hours (k = 3)</b>',
                font: { size: 16, color: colorPalette.navyDark, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
                x: 0.05,
                y: 0.98
            },
            xaxis: {
                title: { text: 'Daily Social Media Hours (hrs/day)', font: { size: 13, color: colorPalette.navyDark } },
                gridcolor: '#e2e8f0',
                zerolinecolor: '#cbd5e1'
            },
            yaxis: {
                title: { text: 'Weekly Study Hours (hrs/week)', font: { size: 13, color: colorPalette.navyDark } },
                gridcolor: '#e2e8f0',
                zerolinecolor: '#cbd5e1'
            },
            plot_bgcolor: '#f8fafc',
            paper_bgcolor: '#ffffff',
            hovermode: 'closest',
            updatemenus: [{
                buttons: kmButtons,
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
            legend: {
                orientation: 'h',
                x: 0.0,
                y: -0.22,
                font: { size: 11, color: '#57606a' }
            },
            margin: { t: 95, r: 25, b: 70, l: 60 }
        };

        Plotly.newPlot(kmContainer, allKmTraces, kmLayout, sharedConfig);
    }
}
