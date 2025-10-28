document.addEventListener('DOMContentLoaded', () => {

    // --- Global Chart Instances ---
    let pastDataChart = null;
    let forecastChart = null;
    let topCasesConsultationChart = null;
    let topCasesDiagnosisChart = null;
    let topCasesMortalityChart = null;

    // --- DOM Elements ---
    const sidebarButtons = document.querySelectorAll('.dashboard-sidebar .nav-button');
    const contentPages = document.querySelectorAll('.dashboard-main-content .page-content');
    
    // Prediction Page
    const predictionForm = document.getElementById('prediction-form');
    const predictionResultBox = document.getElementById('prediction-result');
    const predictionResultContainer = document.getElementById('prediction-result-container');

    // Past Data Page
    const pastDataChartCanvas = document.getElementById('past-data-chart');

    // Forecast Page
    const forecastYearSelect = document.getElementById('forecast-year');
    const forecastChartCanvas = document.getElementById('forecast-chart');

    // Top Cases Page
    // 
    // 🔥 --- FIX 1: ID corrected from 'top-cases-month' to 'topCasesMonthSelect' ---
    //
    const topCasesMonthSelect = document.getElementById('topCasesMonthSelect');
    const topCasesTableConsultation = document.getElementById('top-cases-table-consultation');
    const topCasesChartCanvasConsultation = document.getElementById('top-cases-chart-consultation');
    const topCasesTableDiagnosis = document.getElementById('top-cases-table-diagnosis');
    const topCasesChartCanvasDiagnosis = document.getElementById('top-cases-chart-diagnosis');
    const topCasesTableMortality = document.getElementById('top-cases-table-mortality');
    const topCasesChartCanvasMortality = document.getElementById('top-cases-chart-mortality');

    // --- Utility Functions ---

    function showNotification(message, type = 'success') {
        const notif = document.createElement('div');
        notif.className = `flash ${type}`;
        notif.textContent = message;
        const header = document.querySelector('.header-nav');
        header.insertAdjacentElement('afterend', notif);
        setTimeout(() => {
            notif.style.opacity = '0';
            setTimeout(() => notif.remove(), 300);
        }, 5000);
    }

    function destroyChart(chartInstance) {
        if (chartInstance) {
            chartInstance.destroy();
        }
        return null;
    }

    function populateTopCaseCategory(tableEl, chartCanvasEl, chartInstance, data, barColor = '#007bff') {
        // --- Table ---
        tableEl.innerHTML = `
            <thead><tr>
                <th>Case ID</th>
                <th>Case Name</th>
                <th>Total Cases</th>
            </tr></thead>
        `;
        let tbody = '<tbody>';
        if (data.table.length === 0) {
            tbody += '<tr><td colspan="3">No data for this category.</td></tr>';
        } else {
            data.table.forEach(row => {
                tbody += `<tr>
                    <td>${row.Case}</td>
                    <td>${row.CaseName}</td>
                    <td>${row.Total}</td>
                </tr>`;
            });
        }
        tbody += '</tbody>';
        tableEl.innerHTML += tbody;

        // --- Chart ---
        chartInstance = destroyChart(chartInstance);
        chartInstance = new Chart(chartCanvasEl, {
            type: 'bar',
            data: {
                labels: data.chart_data.labels,
                datasets: [{
                    label: 'Total Cases',
                    data: data.chart_data.data,
                    backgroundColor: barColor
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: { display: data.chart_data.labels.length > 0 }
                }
            }
        });
        return chartInstance;
    }

    // --- Page Functions ---

    // 1️⃣ Prediction
    async function handlePredictionSubmit(event) {
        event.preventDefault();
        predictionResultContainer.style.display = 'none';
        
        const data = {
            year: document.getElementById('pred-year').value,
            month: document.getElementById('pred-month').value,
            consult_type: document.getElementById('pred-consult').value,
            case_id: document.getElementById('pred-case').value,
            sex: document.getElementById('pred-sex').value,
            age_range: document.getElementById('pred-age').value,
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Prediction failed');
            predictionResultBox.textContent = `${result.prediction} estimated total cases.`;
            predictionResultBox.className = 'prediction-result-box success';
            predictionResultContainer.style.display = 'block';
        } catch (error) {
            predictionResultBox.textContent = `Error: ${error.message}`;
            predictionResultBox.className = 'prediction-result-box error';
            predictionResultContainer.style.display = 'block';
        }
    }

    // 2️⃣ Past Data
    async function loadPastData() {
        try {
            const response = await fetch('/api/past_data');
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to fetch past data');

            pastDataChart = destroyChart(pastDataChart);
            pastDataChart = new Chart(pastDataChartCanvas, {
                type: 'line',
                data: {
                    labels: result.chart_data.labels,
                    datasets: [{
                        label: 'Total Cases',
                        data: result.chart_data.data,
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0,123,255,0.1)',
                        fill: true,
                        tension: 0.1
                    }]
                }
            });
        } catch (error) {
            showNotification(error.message, 'error');
        }
    }

    // 3️⃣ Forecast
    async function loadForecast() {
        // Safety check
        if (!forecastYearSelect) return;

        const year = forecastYearSelect.value;
        if (!year) return;
        try {
            const response = await fetch('/api/forecast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ year })
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `Failed to fetch forecast for ${year}`);

            forecastChart = destroyChart(forecastChart);
            forecastChart = new Chart(forecastChartCanvas, {
                type: 'line',
                data: {
                    labels: result.labels,
                    datasets: [
                        {
                            label: `Actual (${year})`,
                            data: result.actual,
                            borderColor: '#28a745',
                            tension: 0.1
                        },
                        {
                            label: `Predicted (${year})`,
                            data: result.predicted,
                            borderColor: '#fd7e14',
                            borderDash: [5, 5],
                            tension: 0.1
                        }
                    ]
                }
            });
        } catch (error) {
            destroyChart(forecastChart);
            showNotification(error.message, 'error');
        }
    }

    // 4️⃣ Top Cases
    async function loadTopCases() {
        // Safety check
        if (!topCasesMonthSelect) return;

        const month = topCasesMonthSelect.value;
        if (!month) return;

        try {
            // Force no-cache + timestamp param to avoid stale responses
            const response = await fetch('/api/top_cases', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache, no-store, must-revalidate'
                },
                body: JSON.stringify({ month: month, t: Date.now() })
            });

            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `Failed to fetch top cases for month ${month}`);

            topCasesConsultationChart = destroyChart(topCasesConsultationChart);
            topCasesDiagnosisChart = destroyChart(topCasesDiagnosisChart);
            topCasesMortalityChart = destroyChart(topCasesMortalityChart);

            topCasesConsultationChart = populateTopCaseCategory(
                topCasesTableConsultation,
                topCasesChartCanvasConsultation,
                topCasesConsultationChart,
                result.consultation,
                '#007bff'
            );
            topCasesDiagnosisChart = populateTopCaseCategory(
                topCasesTableDiagnosis,
                topCasesChartCanvasDiagnosis,
                topCasesDiagnosisChart,
                result.diagnosis,
                '#28a745'
            );
            topCasesMortalityChart = populateTopCaseCategory(
                topCasesTableMortality,
                topCasesChartCanvasMortality,
                topCasesMortalityChart,
                result.mortality,
                '#dc3545'
            );

        } catch (error) {
            topCasesConsultationChart = destroyChart(topCasesConsultationChart);
            topCasesDiagnosisChart = destroyChart(topCasesDiagnosisChart);
            topCasesMortalityChart = destroyChart(topCasesMortalityChart);
            showNotification(error.message, 'error');
        }
    }

    // --- Sidebar Navigation ---
    sidebarButtons.forEach(button => {
        button.addEventListener('click', () => {
            const pageId = button.getAttribute('data-page');
            sidebarButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            contentPages.forEach(page => {
                page.id === `page-${pageId}`
                    ? page.classList.add('active')
                    : page.classList.remove('active');
            });

            // Load corresponding data
            if (pageId === 'past-data') loadPastData();
            else if (pageId === 'forecast') loadForecast();
            else if (pageId === 'top-cases') loadTopCases();
        });
    });

    // --- Populate Year Options ---
    async function populateYears() {
        // Safety check
        if (!forecastYearSelect) return;

        try {
            const response = await fetch('/api/available_years');
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to fetch years');
            forecastYearSelect.innerHTML = '';
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = 'Select a year...';
            placeholder.disabled = true;
            placeholder.selected = true;
            forecastYearSelect.appendChild(placeholder);
            result.years.forEach(year => {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                forecastYearSelect.appendChild(option);
            });
        } catch (error) {
            console.error("Error populating years:", error);
            // Fallback
            const currentYear = new Date().getFullYear();
            for (let year = currentYear + 5; year >= 2020; year--) {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                forecastYearSelect.appendChild(option);
            }
        }
    }

    // --- Event Listeners ---
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    //
    // 🔥 --- FIX 2: Added a safety check in case the element doesn't exist ---
    //
    if (forecastYearSelect) {
        forecastYearSelect.addEventListener('change', loadForecast);
    }
    if (topCasesMonthSelect) {
        topCasesMonthSelect.addEventListener('change', loadTopCases);
    }

    // --- Initial Load ---
    populateYears();
});