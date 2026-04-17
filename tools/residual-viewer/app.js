const state = {
    data: window.VIEWER_DATA || [],
    filter: 'all',
    selectedId: null,
    chartInstance: null,
    overviewChart: null
};

function init() {
    // Sort logic: Absolute residual descending so most deviated are at top
    state.data.sort((a, b) => Math.abs(b.residual) - Math.abs(a.residual));
    document.getElementById('stat-total').textContent = state.data.length;
    
    // Bind filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.filter = e.target.getAttribute('data-filter');
            renderSidebar();
        });
    });

    renderSidebar();
    drawOverviewScatter();
}

function drawOverviewScatter() {
    const ctx = document.getElementById('scatterChart').getContext('2d');
    
    const colors = {
        'A': '#ef4444', 
        'B': '#10b981', 
        'Normal': '#94a3b8'
    };

    const groups = { 'Pattern B (再燃)': [], 'Normal (適正)': [], 'Pattern A (失速)': [] };

    state.data.forEach(item => {
        let gName = 'Normal (適正)';
        if (item.pattern.includes('Pattern A')) gName = 'Pattern A (失速)';
        if (item.pattern.includes('Pattern B')) gName = 'Pattern B (再燃)';

        groups[gName].push({
            x: item.y_pred,
            y: item.y_3m_gt,
            title: item.title_ja,
            residual: item.residual,
            id: item.id
        });
    });

    const datasets = Object.keys(groups).map(key => {
        let color = colors['Normal'];
        if (key.includes('Pattern A')) color = colors['A'];
        if (key.includes('Pattern B')) color = colors['B'];

        return {
            label: key,
            data: groups[key],
            backgroundColor: color,
            borderColor: color,
            pointRadius: 6,
            pointHoverRadius: 8
        };
    });

    state.overviewChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            const pt = ctx.raw;
                            return `${pt.title} (予測: ${pt.x.toFixed(1)}, 実測: ${pt.y.toFixed(1)}, 残差: ${pt.residual.toFixed(1)})`;
                        }
                    }
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            xMin: 0, xMax: 100,
                            yMin: 0, yMax: 100,
                            borderColor: '#cbd5e1',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: { content: 'Perfect Prediction (y=x)', display: true, position: 'end', backgroundColor: '#cbd5e1', color: '#1e293b' }
                        }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: '3ヶ月後 予測値 (Predicted)' }, min: 0, max: 100 },
                y: { title: { display: true, text: '3ヶ月後 実測値 (Actual)' }, min: 0, max: 100 }
            },
            onClick: (e, activeElements) => {
                if (activeElements.length > 0) {
                    const datasetIndex = activeElements[0].datasetIndex;
                    const index = activeElements[0].index;
                    const pt = datasets[datasetIndex].data[index];
                    selectAnime(pt.id);
                }
            }
        }
    });
}

function getPatternGroup(pattern) {
    if (pattern.includes('Pattern A')) return 'A';
    if (pattern.includes('Pattern B')) return 'B';
    return 'Normal';
}

function renderSidebar() {
    const list = document.getElementById('anime-list');
    list.innerHTML = '';
    
    const filtered = state.data.filter(item => {
        if (state.filter === 'all') return true;
        return item.pattern.includes(state.filter);
    });

    filtered.forEach(item => {
        const li = document.createElement('li');
        li.className = `anime-item ${state.selectedId === item.id ? 'selected' : ''}`;
        
        const group = getPatternGroup(item.pattern);
        let resText = item.residual > 0 ? `+${item.residual.toFixed(1)}` : item.residual.toFixed(1);

        li.innerHTML = `
            <div class="anime-item-title">${item.title_ja}</div>
            <div class="anime-item-meta">
                <span class="color-${group}">${item.pattern.split(' ')[0]}</span>
                <span class="meta-residual color-${group}">残差: ${resText}</span>
            </div>
        `;
        li.addEventListener('click', () => selectAnime(item.id));
        list.appendChild(li);
    });
}

function selectAnime(id) {
    state.selectedId = id;
    renderSidebar(); // Update selection highlight
    
    const item = state.data.find(d => d.id === id);
    if (!item) return;

    document.getElementById('welcome-state').classList.add('hidden');
    document.getElementById('detail-state').classList.remove('hidden');

    // Update Header
    document.getElementById('anime-title').textContent = item.title_ja;
    document.getElementById('anime-dates').textContent = `放送開始日: ${item.air_start}`;
    
    const group = getPatternGroup(item.pattern);
    const badge = document.getElementById('anime-pattern');
    badge.textContent = item.pattern;
    badge.className = `badge badge-${group}`;

    // Update Metrics
    document.getElementById('val-pred').textContent = item.y_pred.toFixed(1);
    document.getElementById('val-actual').textContent = item.y_3m_gt.toFixed(1);
    
    let resText = item.residual > 0 ? `+${item.residual.toFixed(1)}` : item.residual.toFixed(1);
    document.getElementById('val-residual').textContent = resText;
    
    const metricCard = document.getElementById('val-residual').closest('.metric-card');
    metricCard.style.backgroundColor = group === 'A' ? '#ef4444' : (group === 'B' ? '#10b981' : '#64748b');
    metricCard.style.borderColor = metricCard.style.backgroundColor;

    drawChart(item);
}

function drawChart(item) {
    const ctx = document.getElementById('mainChart').getContext('2d');
    
    if (state.chartInstance) {
        state.chartInstance.destroy();
    }

    const labels = item.timeseries.map(d => `Day ${d.day_index}`);
    const dataGT = item.timeseries.map(d => d.gt);
    const dataPV = item.timeseries.map(d => d.pv);

    state.chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Google Trends スコア',
                    data: dataGT,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    yAxisID: 'ygt',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHitRadius: 10,
                    fill: true
                },
                {
                    label: 'Wikipedia PV',
                    data: dataPV,
                    borderColor: '#3b82f6',
                    backgroundColor: 'transparent',
                    yAxisID: 'ypv',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHitRadius: 10
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.9)' },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            xMin: 21, xMax: 21,
                            borderColor: '#94a3b8',
                            borderWidth: 2,
                            borderDash: [4, 4],
                            label: { content: '初速データの境界 (Day 21)', display: true, position: 'start', backgroundColor: '#94a3b8', font: {size: 10} }
                        },
                        line2: {
                            type: 'line',
                            xMin: 90, xMax: 90,
                            borderColor: '#ef4444',
                            borderWidth: 2,
                            borderDash: [4, 4],
                            label: { content: '3M 予測ターゲット (Day 90)', display: true, position: 'start', backgroundColor: '#ef4444', font: {size: 10} }
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 15 }
                },
                ygt: {
                    type: 'linear',
                    position: 'right',
                    min: 0, max: 100,
                    title: { display: true, text: 'Google Trends' },
                    grid: { display: false }
                },
                ypv: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    title: { display: true, text: 'Wikipedia PV' },
                    grid: { color: '#f1f5f9' }
                }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', init);
