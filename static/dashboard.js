document.addEventListener('DOMContentLoaded', () => {
            // --- Global Chart Instances ---
            let charts = {
                pastData: null,
                consultation: null,
                diagnosis: null,
                mortality: null,
                featureImportance: null
            };
            let latestPredictedVisits = null;


            // --- Sidebar Navigation Logic ---
            const sidebarButtons = document.querySelectorAll('.nav-button');
            const pageContents = document.querySelectorAll('.page-content');

            sidebarButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const pageId = button.getAttribute('data-page');

                    // Set active button
                    sidebarButtons.forEach(btn => btn.classList.remove('active'));
                    button.classList.add('active');

                    // Show active page
                    pageContents.forEach(page => {
                        if (page.id === `page-${pageId}`) {
                            page.classList.add('active');
                        } else {
                            page.classList.remove('active');
                        }
                    });

                    // Load data for the activated page
                    if (pageId === 'past-data') loadPastData();
                    if (pageId === 'top-cases') loadTopCases();       
                    if (pageId === 'model-insights') loadModelInsights();
                    if (pageId === 'resource-management') {
    const month = parseInt(resourceMonthSelect.value, 10) || 1;
    loadResourceForMonth(month);
}

                });
            });

            // --- Error Handling ---
            function showError(elementId, message) {
                const el = document.getElementById(elementId);
                el.textContent = message;
                el.style.display = 'block';
            }
            function hideError(elementId) {
                document.getElementById(elementId).style.display = 'none';
            }

            // --- 1. Past Data Logic ---
            async function loadPastData() {
                try {
                    const response = await fetch('/api/past_data');
                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.error || 'Failed to fetch past data.');
                    }
                    const data = await response.json();
                    hideError('past-data-error');
                    
                    if (charts.pastData) charts.pastData.destroy();
                    const ctx = document.getElementById('past-data-chart').getContext('2d');
                    charts.pastData = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.chart_data.labels,
                            datasets: [{
                                label: 'Total Patients',
                                data: data.chart_data.data,
                                borderColor: 'rgb(59, 130, 246)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                fill: true,
                                tension: 0.1
                            }]
                        },
                        options: { 
                            responsive: true, 
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } }
                        }
                    });
                } catch (error) {
                    console.error('Error loading past data:', error);
                    showError('past-data-error', error.message);
                }
            }
// --- 2. Top Cases Logic ---
const monthSelect = document.getElementById('topCasesMonthSelect');
const resourceMonthSelect = document.getElementById('resourceMonthSelect');
monthSelect.addEventListener('change', () => {
    // When month changes, refresh Top Cases + Resource Needs
    loadTopCases();
});

resourceMonthSelect.addEventListener('change', () => {
    const month = parseInt(resourceMonthSelect.value, 10);
    loadResourceForMonth(month);
});



function updateResourceTable(resources) {
    const table = document.getElementById('resource-needs-table');
    if (!table) return;

    if (!resources || !resources.length) {
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Resource</th>
                    <th>Monthly Demand</th>
                    <th>Monthly Capacity</th>
                    <th>Utilization (%)</th>
                    <th>Status</th>
                    <th>Shortage</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colspan="6" style="text-align:center;">No resource data available.</td>
                </tr>
            </tbody>
        `;
        return;
    }

    table.innerHTML = `
        <thead>
            <tr>
                <th>Resource</th>
                <th>Monthly Demand</th>
                <th>Monthly Capacity</th>
                <th>Utilization (%)</th>
                <th>Status</th>
                <th>Shortage</th>
            </tr>
        </thead>
        <tbody>
            ${resources.map(r => `
                <tr class="${r.status}">
                    <td>${r.resource_name}</td>
                    <td>${r.monthly_demand} ${r.unit || ''}</td>
                    <td>${r.monthly_capacity} ${r.unit || ''}</td>
                    <td>${r.utilization.toFixed ? r.utilization.toFixed(1) : r.utilization}</td>
                    <td>${r.status}</td>
                    <td>${r.shortage} ${r.unit || ''}</td>
                </tr>
            `).join('')}
        </tbody>
    `;
}

async function loadTopCases() {
    const month = monthSelect.value;
    try {
        hideError('top-cases-error');
        const response = await fetch('/api/top_cases', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ month: parseInt(month) })
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Failed to fetch top cases data.');
        }
        const data = await response.json();

        // 🔹 1) Save predicted visits from the backend
        latestPredictedVisits = data.total_summary && data.total_summary.predicted_total
            ? data.total_summary.predicted_total
            : null;

        // 🔹 2) Update Consultation / Diagnosis / Mortality sections
        updateTopCaseSection('consultation', data.consultation);
        updateTopCaseSection('diagnosis', data.diagnosis);
        updateTopCaseSection('mortality', data.mortality);

// 🔹 3) Also refresh Resource Management for the same month (optional)
const monthInt = parseInt(monthSelect.value, 10);
if (!isNaN(monthInt)) {
    loadResourceForMonth(monthInt);
}


    } catch (error) {
        console.error('Error loading top cases:', error);
        showError('top-cases-error', error.message);
    }
}

    async function loadResourceForMonth(month) {
        try {
            hideError('resource-error');
            const summaryLabel = document.getElementById('resource-summary');
            if (summaryLabel) {
                summaryLabel.textContent = 'Loading predicted volume and resource needs...';
            }

            // --- Step 1: get predicted_total for this month ---
            const topCasesRes = await fetch('/api/top_cases', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ month: month })
            });

            const topCasesData = await topCasesRes.json();
            if (!topCasesRes.ok || topCasesData.error) {
                throw new Error(topCasesData.error || 'Failed to fetch top cases data for resources.');
            }

            const predictedTotal = topCasesData.total_summary?.predicted_total || 0;
            latestPredictedVisits = predictedTotal;

            if (!predictedTotal || predictedTotal <= 0) {
                throw new Error('No predicted volume available for this month.');
            }

            if (summaryLabel) {
                summaryLabel.textContent =
                    `Predicted patient volume for this month: ${Math.round(predictedTotal).toLocaleString()} visits.`;
            }

            // --- Step 2: call resource_needs using that predictedTotal ---
            const resNeedsRes = await fetch('/api/resource_needs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    predicted_visits: predictedTotal,
                    general_factor: 1.0,
                    nurses_per_shift: 7
                })
            });

            const resNeedsData = await resNeedsRes.json();
            if (!resNeedsRes.ok || resNeedsData.error) {
                throw new Error(resNeedsData.error || 'Failed to fetch resource needs.');
            }

            updateResourceTable(resNeedsData.resources);

        } catch (error) {
            console.error('Error loading resources for month:', error);
            showError('resource-error', error.message);
            const summaryLabel = document.getElementById('resource-summary');
            if (summaryLabel) summaryLabel.textContent = '';
            updateResourceTable([]); // clear table on error
        }
    }

            function updateTopCaseSection(type, data) {
                // Update Chart
                const chartLabels = data.chart_data.labels;
                const chartData = data.chart_data.data;

                if (charts[type]) charts[type].destroy();
                const ctx = document.getElementById(`top-cases-chart-${type}`).getContext('2d');
                charts[type] = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: chartLabels,
                        datasets: [{
                            label: 'Predicted Total',
                            data: chartData,
                            backgroundColor: 'rgba(59, 130, 246, 0.7)'
                        }]
                    },
                    options: { 
                        responsive: true,
                        indexAxis: 'y', // Horizontal bar chart
                        plugins: { legend: { display: false } }
                    }
                });

                // Update Table
                const table = document.getElementById(`top-cases-table-${type}`);
                table.innerHTML = `
                    <thead>
                        <tr>
                            <th>Case Name</th>
                            <th>Predicted Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.table.map(row => `
                            <tr>
                                <td>${row.CaseName}</td>
                                <td>${Math.round(row.Total)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                `;
            }

            // --- 3. Model Insights Logic (NEW) ---
            async function loadModelInsights() {
                try {
                    const response = await fetch('/api/feature_importance');
                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.error || 'Failed to fetch model insights.');
                    }
                    const data = await response.json(); // e.g., [{"Feature": "Age_range", "Importance": 1.526}, ...]
                    hideError('insights-error');

                    // Data is already sorted by importance in the training script
                    const labels = data.map(row => row.Feature);
                    const scores = data.map(row => row.Importance);

                    // Update Table
                    const table = document.getElementById('feature-importance-table');
                    table.innerHTML = `
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Importance Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.map(row => `
                                <tr>
                                    <td>${row.Feature}</td>
                                    <td>${row.Importance.toFixed(4)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    `;

                    // Update Chart
                    if (charts.featureImportance) charts.featureImportance.destroy();
                    const ctx = document.getElementById('feature-importance-chart').getContext('2d');
                    charts.featureImportance = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Importance Score',
                                data: scores,
                                backgroundColor: [
                                    'rgba(59, 130, 246, 0.7)',
                                    'rgba(16, 185, 129, 0.7)',
                                    'rgba(245, 158, 11, 0.7)',
                                    'rgba(239, 68, 68, 0.7)',
                                    'rgba(139, 92, 246, 0.7)',
                                    'rgba(107, 114, 128, 0.7)'
                                ],
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            indexAxis: 'y', // Make it horizontal
                            plugins: {
                                legend: { display: false },
                                title: { display: true, text: 'Model Feature Importance' }
                            },
                            scales: {
                                x: {
                                    title: { display: true, text: 'Importance Score' }
                                }
                            }
                        }
                    });

                } catch (error) {
                    console.error('Error loading model insights:', error);
                    showError('insights-error', error.message);
                }
            }

            // --- Initial Load ---
            // Load the data for the default active tab
            loadPastData();
            loadModelInsights();});