document.addEventListener('DOMContentLoaded', () => {

    // --- Global Chart Instances ---
    let pastDataChart = null;
    let forecastChart = null;
    let topCasesChart = null;

    // --- DOM Elements ---
    const sidebarButtons = document.querySelectorAll('.dashboard-sidebar .nav-button');
    const contentPages = document.querySelectorAll('.dashboard-main-content .page-content');
    
    // Prediction Page
    const predictionForm = document.getElementById('prediction-form');
    const predictionResultBox = document.getElementById('prediction-result');
    const predictionResultContainer = document.getElementById('prediction-result-container');

    // Past Data Page
    const pastDataTable = document.getElementById('past-data-table');
    const pastDataChartCanvas = document.getElementById('past-data-chart');

    // Forecast Page
    const forecastYearSelect = document.getElementById('forecast-year');
    const forecastChartCanvas = document.getElementById('forecast-chart');

    // Top Cases Page
    const topCasesMonthSelect = document.getElementById('top-cases-month');
    const topCasesTable = document.getElementById('top-cases-table');
    const topCasesChartCanvas = document.getElementById('top-cases-chart');

    // --- Utility Functions ---

    /**
     * Shows a notification-style message.
     * @param {string} message - The message to display.
     * @param {string} type - 'success' or 'error'.
     */
    function showNotification(message, type = 'success') {
        const notif = document.createElement('div');
        notif.className = `flash ${type}`;
        notif.textContent = message;
        
        // Insert after the header
        const header = document.querySelector('.header-nav');
        header.insertAdjacentElement('afterend', notif);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            notif.style.opacity = '0';
            setTimeout(() => notif.remove(), 300);
        }, 5000);
    }

    /**
     * Clears any existing chart instance.
     * @param {Chart} chartInstance - The Chart.js instance to destroy.
     * @returns {null}
     */
    function destroyChart(chartInstance) {
        if (chartInstance) {
            chartInstance.destroy();
        }
        return null;
    }

    // --- Page Load/Event Functions ---

    // 1. Prediction Page
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

            if (!response.ok) {
                throw new Error(result.error || 'Prediction failed');
            }

            predictionResultBox.textContent = `${result.prediction} estimated total cases.`;
            predictionResultBox.className = 'prediction-result-box success';
            predictionResultContainer.style.display = 'block';

        } catch (error) {
            predictionResultBox.textContent = `Error: ${error.message}`;
            predictionResultBox.className = 'prediction-result-box error';
            predictionResultContainer.style.display = 'block';
        }
    }

    // 2. Past Data Page
    async function loadPastData() {
        try {
            const response = await fetch('/api/past_data');
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Failed to fetch past data');
            }
            

            // b) Populate Chart
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

    // 3. Forecast Page
    async function loadForecast() {
        const year = forecastYearSelect.value;
        if (!year) return;

        try {
            const response = await fetch('/api/forecast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ year: year })
            });

            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || `Failed to fetch forecast for ${year}`);
            }

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

    // 4. Top Cases Page
    async function loadTopCases() {
        const month = topCasesMonthSelect.value;

        try {
            const response = await fetch('/api/top_cases', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ month: month })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || `Failed to fetch top cases for month ${month}`);
            }

            // a) Populate Table
            topCasesTable.innerHTML = `
                <thead><tr>
                    <th>Case ID</th>
                    <th>Case Name</th>
                    <th>Total Cases</th>
                </tr></thead>
            `;
            let tbody = '<tbody>';
            result.table.forEach(row => {
                tbody += `<tr>
                    <td>${row.Case}</td>
                    <td>${row.CaseName}</td>
                    <td>${row.Total}</td>
                </tr>`;
            });
            tbody += '</tbody>';
            topCasesTable.innerHTML += tbody;

            // b) Populate Chart
            topCasesChart = destroyChart(topCasesChart);
            topCasesChart = new Chart(topCasesChartCanvas, {
                type: 'bar',
                data: {
                    labels: result.chart_data.labels,
                    datasets: [{
                        label: 'Total Cases',
                        data: result.chart_data.data,
                        backgroundColor: '#007bff'
                    }]
                },
                options: {
                    indexAxis: 'y', // Horizontal bar chart
                    responsive: true
                }
            });

        } catch (error) {
            destroyChart(topCasesChart);
            showNotification(error.message, 'error');
        }
    }

    // --- Tab Switching Logic ---
    sidebarButtons.forEach(button => {
        button.addEventListener('click', () => {
            const pageId = button.getAttribute('data-page');

            // 1. Update button active state
            sidebarButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // 2. Show the correct page
            contentPages.forEach(page => {
                if (page.id === `page-${pageId}`) {
                    page.classList.add('active');
                } else {
                    page.classList.remove('active');
                }
            });

            // 3. Load data for the activated page
            // (Don't re-load 'predict' page)
            if (pageId === 'past-data') {
                loadPastData();
            } else if (pageId === 'forecast') {
                loadForecast();
            } else if (pageId === 'top-cases') {
                loadTopCases();
            }
        });
    });

    // --- Initializers ---

    // Populate Year dropdown for forecast
// Populate Year dropdown for forecast
    async function populateYears() {
        try {
            const response = await fetch('/api/available_years');
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Failed to fetch years');
            }

            // Clear any existing options
            forecastYearSelect.innerHTML = '';
            
            // Add a default placeholder
            const placeholder = document.createElement('option');
            placeholder.value = ''; // Empty value so the check 'if (!year) return' works
            placeholder.textContent = 'Select a year...';
            placeholder.disabled = true;
            placeholder.selected = true;
            forecastYearSelect.appendChild(placeholder);

            // Populate with years from the API
            result.years.forEach(year => {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                forecastYearSelect.appendChild(option);
            });

        } catch (error) {
            console.error("Error populating years:", error);
            // Fallback to default if API fails
            const currentYear = new Date().getFullYear();
            for (let year = currentYear + 5; year >= 2020; year--) {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                forecastYearSelect.appendChild(option);
            }
        }
    }
    
    // Add event listeners for dynamic selects
    predictionForm.addEventListener('submit', handlePredictionSubmit);
    forecastYearSelect.addEventListener('change', loadForecast);
    topCasesMonthSelect.addEventListener('change', loadTopCases);

    // --- Initial Page Load ---
    populateYears();
    // Manually trigger the load for the default visible page ('predict' has no data load)
    // If you change the default, trigger its load function here.
    // e.g., if 'past-data' was default: loadPastData();
});
