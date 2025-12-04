/**
 * 🚦 Digital Twin City Simulation - Frontend Application
 * ========================================================
 * 
 * Interactive map visualization with GNN traffic prediction
 * 
 * Author: Digital Twin City Simulation Team
 * Date: December 2025
 */

// ============================================================
// CONFIGURATION
// ============================================================

const CONFIG = {
    API_BASE: 'http://localhost:5000/api',
    // Pune, India - center coordinates
    MAP_CENTER: [18.5204, 73.8567],
    MAP_ZOOM: 13,
    // Set to true to overlay on real map, false for abstract view
    USE_REAL_MAP: true,
    // Scale factor to convert graph coordinates to real-world offset
    COORD_SCALE: 0.001,  // Adjust this to fit city size
    COLORS: {
        road: '#3498db',
        roadHover: '#2980b9',
        // Metro line colors
        metroRed: '#FF0000',
        metroBlue: '#0000FF', 
        metroGreen: '#00FF00',
        highCongestion: '#e74c3c',  // Red for high congestion
        lowCongestion: '#2ecc71',   // Green for low congestion
        closed: '#ff69b4',          // Pink for blocked roads,
        // Node colors
        node: '#95a5a6',
        nodeMetro: '#9b59b6',
        // Amenity colors
        hospital: '#e74c3c',
        park: '#27ae60',
        school: '#f39c12',
        mall: '#e91e63',
        factory: '#795548',
        warehouse: '#607d8b',
        office: '#3498db',
        community_center: '#9c27b0'
    }
};

// ============================================================
// STATE
// ============================================================

const state = {
    map: null,
    graphData: null,
    predictions: null,
    baselinePredictions: null,  // Store baseline for comparison
    baselineStats: null,
    closedRoads: new Set(),
    layers: {
        roads: null,
        metro: null,
        nodes: null,
        amenities: null
    },
    visible: {
        roads: true,
        metro: true,
        nodes: true,
        amenities: true
    }
};

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', init);

async function init() {
    showLoading('Initializing...');
    
    // Initialize map
    initMap();
    
    // Setup event listeners
    setupEventListeners();
    
    // Check API status
    await checkStatus();
    
    // Load graph data
    await loadGraphData();
    
    hideLoading();
}

function initMap() {
    if (CONFIG.USE_REAL_MAP) {
        // Real map with OpenStreetMap tiles
        state.map = L.map('map', {
            center: CONFIG.MAP_CENTER,
            zoom: CONFIG.MAP_ZOOM,
            zoomControl: true,
            zoomDelta: 0.5,          // Smaller zoom steps
            zoomSnap: 0.5,           // Snap to 0.5 increments
            wheelPxPerZoomLevel: 120 // More pixels needed to zoom (slower scroll zoom)
        });

        // Add OpenStreetMap tile layer (dark theme)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(state.map);
    } else {
        // Abstract view with simple coordinates
        state.map = L.map('map', {
            center: [0, 0],
            zoom: 2,
            crs: L.CRS.Simple,
            zoomControl: true,
            zoomDelta: 0.5,
            zoomSnap: 0.5,
            wheelPxPerZoomLevel: 120
        });
    }

    // Add zoom control to top-right
    state.map.zoomControl.setPosition('topright');
}

function setupEventListeners() {
    // Add null checks for all elements before adding event listeners
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', runPrediction);
    }
    
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetSimulation);
    }
    
    const viewAnalysisBtn = document.getElementById('view-analysis-btn');
    if (viewAnalysisBtn) {
        viewAnalysisBtn.addEventListener('click', () => {
            window.open('analysis.html', '_blank');
        });
    }
    
    const ctmInitBtn = document.getElementById('ctm-init-btn');
    if (ctmInitBtn) {
        console.log('✅ CTM Init button found');
        ctmInitBtn.addEventListener('click', initCTM);
    } else {
        console.error('❌ CTM Init button NOT found');
    }
    
    const ctmStepBtn = document.getElementById('ctm-step-btn');
    if (ctmStepBtn) {
        console.log('✅ CTM Step button found');
        ctmStepBtn.addEventListener('click', () => stepCTM(1));
    } else {
        console.error('❌ CTM Step button NOT found');
    }
    
    const ctmRunBtn = document.getElementById('ctm-run-btn');
    if (ctmRunBtn) {
        console.log('✅ CTM Run button found');
        ctmRunBtn.addEventListener('click', () => stepCTM(10));
    } else {
        console.error('❌ CTM Run button NOT found');
    }
    
    const ctmResetBtn = document.getElementById('ctm-reset-btn');
    if (ctmResetBtn) {
        console.log('✅ CTM Reset button found');
        ctmResetBtn.addEventListener('click', resetCTM);
    } else {
        console.error('❌ CTM Reset button NOT found');
    }
    
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportData);
    }
    
    // Add more null checks for any other elements with event listeners
    const closeRoadBtn = document.getElementById('close-road-btn');
    if (closeRoadBtn) {
        closeRoadBtn.addEventListener('click', handleCloseRoad);
    }
    
    const reopenRoadBtn = document.getElementById('reopen-road-btn');
    if (reopenRoadBtn) {
        reopenRoadBtn.addEventListener('click', handleReopenRoad);
    }
    
    // Layer toggle checkboxes - FIX: These were missing!
    const layerRoads = document.getElementById('layer-roads');
    if (layerRoads) {
        layerRoads.addEventListener('change', (e) => {
            state.visible.roads = e.target.checked;
            updateLayerVisibility();
        });
    }
    
    const layerMetro = document.getElementById('layer-metro');
    if (layerMetro) {
        layerMetro.addEventListener('change', (e) => {
            state.visible.metro = e.target.checked;
            updateLayerVisibility();
        });
    }
    
    const layerNodes = document.getElementById('layer-nodes');
    if (layerNodes) {
        layerNodes.addEventListener('change', (e) => {
            state.visible.nodes = e.target.checked;
            updateLayerVisibility();
        });
    }
    
    const layerAmenities = document.getElementById('layer-amenities');
    if (layerAmenities) {
        layerAmenities.addEventListener('change', (e) => {
            state.visible.amenities = e.target.checked;
            updateLayerVisibility();
        });
    }
    
    // Clear closures button
    const clearClosuresBtn = document.getElementById('btn-clear-closures');
    if (clearClosuresBtn) {
        clearClosuresBtn.addEventListener('click', clearClosures);
    }
    
    // Random road buttons
    const randomCloseBtn = document.getElementById('btn-random-close');
    if (randomCloseBtn) {
        randomCloseBtn.addEventListener('click', randomCloseRoad);
    }
    
    const randomOpenBtn = document.getElementById('btn-random-open');
    if (randomOpenBtn) {
        randomOpenBtn.addEventListener('click', randomOpenRoad);
    }
    
    // Info panel close button - FIX: This was missing!
    const closeInfoBtn = document.getElementById('close-info');
    if (closeInfoBtn) {
        closeInfoBtn.addEventListener('click', () => {
            const panel = document.getElementById('info-panel');
            if (panel) {
                panel.classList.remove('visible');
            }
        });
    }
}

// Navigate to analysis page with current state
function goToAnalysis() {
    if (state.closedRoads.size === 0) {
        showToast('Block some roads first before viewing analysis!', 'error');
        return;
    }
    
    // Save current state to localStorage
    const analysisData = {
        closedRoads: Array.from(state.closedRoads),
        timestamp: Date.now()
    };
    localStorage.setItem('cityAnalysisData', JSON.stringify(analysisData));
    
    showToast('Opening analysis page...', 'info');
    
    // Navigate to analysis page
    setTimeout(() => {
        window.location.href = 'analysis.html';
    }, 300);
}

// ============================================================
// API CALLS
// ============================================================

async function checkStatus() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/status`);
        const data = await response.json();
        
        updateStatusUI(data);
        return data;
    } catch (error) {
        console.error('Status check failed:', error);
        updateStatusUI({ status: 'offline' });
        showToast('Failed to connect to backend', 'error');
        return null;
    }
}

async function loadGraphData() {
    showLoading('Loading city graph...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/graph`);
        state.graphData = await response.json();
        
        // Debug: count metro edges
        const metroEdges = state.graphData.edges.filter(e => e.is_metro);
        console.log(`Loaded ${state.graphData.node_count} nodes, ${state.graphData.edge_count} edges`);
        console.log(`Metro edges: ${metroEdges.length}`);
        if (metroEdges.length > 0) {
            console.log('Sample metro edge:', metroEdges[0]);
        }
        
        // Render the graph
        renderGraph();
        
        showToast(`Loaded ${state.graphData.node_count} nodes`, 'success');
    } catch (error) {
        console.error('Failed to load graph:', error);
        showToast('Failed to load graph data', 'error');
    }
}

async function runPrediction() {
    showLoading('Running prediction...');
    
    try {
        // FIRST: Get baseline prediction (no closures) for comparison
        const baselineResponse = await fetch(`${CONFIG.API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ closed_roads: [] })
        });
        state.baselinePredictions = await baselineResponse.json();
        state.baselineStats = state.baselinePredictions?.stats || null;
        
        // THEN: Get prediction with current road closures
        const response = await fetch(`${CONFIG.API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                closed_roads: Array.from(state.closedRoads)
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        state.predictions = data;
        
        // Update visualization
        updatePredictionVisualization();
        const baselineStats = state.closedRoads.size > 0 ? state.baselineStats : null;
        updateStatsUI(data.stats, baselineStats);
        
        // Show result message and save data for analysis page
        if (state.closedRoads.size > 0) {
            console.log('=== PREDICTION WITH ROAD CLOSURES ===');
            console.log(`Closed roads: ${Array.from(state.closedRoads).join(', ')}`);
            
            // Save data for analysis page
            const analysisData = {
                closedRoads: Array.from(state.closedRoads),
                baseline: state.baselinePredictions,
                withClosures: state.predictions,
                timestamp: new Date().toISOString()
            };
            
            console.log('💾 Saving analysis data:', {
                closedRoads: analysisData.closedRoads.length,
                baselinePredictions: analysisData.baseline?.predictions?.length || 0,
                withClosuresPredictions: analysisData.withClosures?.predictions?.length || 0,
                baselineStats: analysisData.baseline?.stats,
                withClosuresStats: analysisData.withClosures?.stats
            });
            
            localStorage.setItem('cityAnalysisData', JSON.stringify(analysisData));
            
            showToast(`Prediction done! Click "View Analysis" to see detailed impact`, 'success');
            
            // Show analysis button
            const analysisBtn = document.getElementById('view-analysis-btn');
            if (analysisBtn) analysisBtn.style.display = 'block';
        } else {
            showToast('Prediction done! Showing current traffic levels', 'success');
            const analysisBtn = document.getElementById('view-analysis-btn');
            if (analysisBtn) analysisBtn.style.display = 'none';
        }
    } catch (error) {
        console.error('Prediction failed:', error);
        showToast('Prediction failed: ' + error.message, 'error');
    }
    
    hideLoading();
}

// ============================================================
// RENDERING
// ============================================================

function renderGraph() {
    if (!state.graphData) return;
    
    const { nodes, edges } = state.graphData;
    
    // Calculate bounds of graph coordinates
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    const nodeMap = {};
    nodes.forEach(node => {
        nodeMap[node.id] = node;
        minX = Math.min(minX, node.x);
        maxX = Math.max(maxX, node.x);
        minY = Math.min(minY, node.y);
        maxY = Math.max(maxY, node.y);
    });
    
    // Calculate center and range of graph
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    
    // Transform function: maps graph coords to real-world lat/lng
    let transform;
    if (CONFIG.USE_REAL_MAP) {
        // Map to real Pune coordinates
        // Scale to cover roughly 5km x 5km area (0.045 degrees ~ 5km)
        const scaleFactor = 0.045 / Math.max(rangeX, rangeY);
        transform = (x, y) => [
            CONFIG.MAP_CENTER[0] + (y - centerY) * scaleFactor,
            CONFIG.MAP_CENTER[1] + (x - centerX) * scaleFactor
        ];
    } else {
        // Simple scaling for abstract view
        const scale = 10;
        transform = (x, y) => [y * scale, x * scale];
    }
    
    // Clear existing layers
    Object.values(state.layers).forEach(layer => {
        if (layer) state.map.removeLayer(layer);
    });
    
    // Create layer groups
    state.layers.roads = L.layerGroup();
    state.layers.metro = L.layerGroup();
    state.layers.nodes = L.layerGroup();
    state.layers.amenities = L.layerGroup();
    
    // Render edges
    edges.forEach(edge => {
        const source = nodeMap[edge.source];
        const target = nodeMap[edge.target];
        
        if (!source || !target) return;
        
        const latlngs = [
            transform(source.x, source.y),
            transform(target.x, target.y)
        ];
        
        const isMetro = edge.is_metro === true;  // Strict boolean check
        const isClosed = state.closedRoads.has(`${edge.source}-${edge.target}`);
        
        let color = CONFIG.COLORS.road;  // Default: blue (#3498db)
        let weight = 2;
        let dashArray = null;
        
        if (isMetro) {
            // Metro lines - use their specific colors
            if (edge.line_color) {
                color = edge.line_color;
            } else {
                const line = (edge.metro_line || '').toLowerCase();
                if (line.includes('red')) color = CONFIG.COLORS.metroRed;
                else if (line.includes('blue')) color = CONFIG.COLORS.metroBlue;
                else if (line.includes('green')) color = CONFIG.COLORS.metroGreen;
                else color = '#9b59b6';  // Default purple
            }
            weight = 5;
        }
        
        if (isClosed) {
            color = CONFIG.COLORS.closed;
            dashArray = '8, 8';
        }
        
        const polyline = L.polyline(latlngs, {
            color: color,
            weight: weight,
            opacity: isMetro ? 0.9 : 0.7,
            dashArray: dashArray
        });
        
        // Store edge data for CTM updates
        polyline.feature = {
            source: edge.source,
            target: edge.target,
            is_metro: isMetro
        };
        
        // Add click handler for road closure (not for metro)
        if (!isMetro) {
            polyline.on('click', () => toggleRoadClosure(edge));
            polyline.on('mouseover', function() {
                this.setStyle({ weight: weight + 2, opacity: 1 });
            });
            polyline.on('mouseout', function() {
                this.setStyle({ weight: weight, opacity: 0.7 });
            });
        }
        
        // Add to appropriate layer
        if (isMetro) {
            state.layers.metro.addLayer(polyline);
        } else {
            state.layers.roads.addLayer(polyline);
        }
        
        // Store reference for updates
        edge._polyline = polyline;
    });
    
    // Render nodes
    nodes.forEach(node => {
        const pos = transform(node.x, node.y);
        
        let color = CONFIG.COLORS.node;
        let radius = 3;
        let isAmenity = false;
        let emoji = null;
        const amenity = (node.amenity || 'none').toLowerCase();
        
        // Determine emoji and properties based on amenity type
        if (amenity.includes('hospital')) {
            emoji = '🏥';
            isAmenity = true;
        } else if (amenity.includes('park')) {
            emoji = '🌳';
            isAmenity = true;
        } else if (amenity.includes('school')) {
            emoji = '🏫';
            isAmenity = true;
        } else if (amenity.includes('mall')) {
            emoji = '🛒';
            isAmenity = true;
        } else if (amenity.includes('factory')) {
            emoji = '🏭';
            isAmenity = true;
        } else if (amenity.includes('warehouse')) {
            emoji = '📦';
            isAmenity = true;
        } else if (amenity.includes('office')) {
            emoji = '🏢';
            isAmenity = true;
        } else if (amenity.includes('community')) {
            emoji = '🏛️';
            isAmenity = true;
        }
        
        // Metro stations get special treatment
        if (amenity.includes('metro_station') || node.is_metro) {
            emoji = '🚇';
            isAmenity = true;
        }
        
        let marker;
        
        if (emoji) {
            // Use emoji marker for amenities
            const emojiIcon = L.divIcon({
                html: `<div class="emoji-marker">${emoji}</div>`,
                className: 'emoji-icon',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            });
            marker = L.marker(pos, { icon: emojiIcon });
            
            // Add tooltip for amenities
            marker.bindTooltip(`${amenity.replace('_', ' ').toUpperCase()}`, {
                permanent: false,
                direction: 'top'
            });
        } else {
            // Use circle for regular nodes
            marker = L.circleMarker(pos, {
                radius: radius,
                fillColor: color,
                fillOpacity: 0.5,
                color: color,
                weight: 1
            });
        }
        
        marker.on('click', () => showNodeInfo(node));
        
        if (isAmenity) {
            state.layers.amenities.addLayer(marker);
        }
        state.layers.nodes.addLayer(marker);
    });
    
    // Add visible layers to map (amenities ON by default now)
    if (state.visible.roads) state.layers.roads.addTo(state.map);
    if (state.visible.metro) state.layers.metro.addTo(state.map);
    if (state.visible.nodes) state.layers.nodes.addTo(state.map);
    state.layers.amenities.addTo(state.map);  // Always show amenities
    
    // Fit bounds to show all nodes
    if (CONFIG.USE_REAL_MAP) {
        // Create bounds from transformed node positions
        const allPositions = nodes.map(n => transform(n.x, n.y));
        const bounds = L.latLngBounds(allPositions);
        state.map.fitBounds(bounds, { padding: [50, 50] });
    } else {
        const scale = 10;
        const bounds = [
            [minY * scale, minX * scale],
            [maxY * scale, maxX * scale]
        ];
        state.map.fitBounds(bounds, { padding: [50, 50] });
    }
}

function updateLayerVisibility() {
    // Roads
    if (state.visible.roads && state.layers.roads) {
        state.layers.roads.addTo(state.map);
    } else if (state.layers.roads) {
        state.map.removeLayer(state.layers.roads);
    }
    
    // Metro
    if (state.visible.metro && state.layers.metro) {
        state.layers.metro.addTo(state.map);
    } else if (state.layers.metro) {
        state.map.removeLayer(state.layers.metro);
    }
    
    // Nodes
    if (state.visible.nodes && state.layers.nodes) {
        state.layers.nodes.addTo(state.map);
    } else if (state.layers.nodes) {
        state.map.removeLayer(state.layers.nodes);
    }
    
    // Amenities
    if (state.visible.amenities && state.layers.amenities) {
        state.layers.amenities.addTo(state.map);
    } else if (state.layers.amenities) {
        state.map.removeLayer(state.layers.amenities);
    }
}

function updatePredictionVisualization() {
    if (!state.predictions || !state.graphData) {
        console.warn('Cannot update prediction visualization: missing data');
        return;
    }
    
    const predictions = state.predictions.predictions;
    if (!predictions || predictions.length === 0) {
        console.warn('No predictions to visualize');
        return;
    }
    
    console.log(`Visualizing ${predictions.length} predictions`);
    
    const predMap = {};
    predictions.forEach(p => {
        predMap[`${p.source}-${p.target}`] = p.congestion;
    });
    
    // Build baseline map for comparison (only if we have closed roads)
    const baselineMap = {};
    const hasClosures = state.closedRoads.size > 0;
    
    if (state.baselinePredictions && hasClosures) {
        state.baselinePredictions.predictions.forEach(p => {
            baselineMap[`${p.source}-${p.target}`] = p.congestion;
        });
    }
    
    // Calculate percentiles for congestion levels
    const congestionValues = predictions.map(p => p.congestion);
    const sorted = [...congestionValues].sort((a, b) => a - b);
    const p20 = sorted[Math.floor(sorted.length * 0.20)];
    const p40 = sorted[Math.floor(sorted.length * 0.40)];
    const p60 = sorted[Math.floor(sorted.length * 0.60)];
    const p80 = sorted[Math.floor(sorted.length * 0.80)];
    
    // Track changed edges
    let increasedCount = 0;
    let decreasedCount = 0;
    let visualizedCount = 0;
    
    state.graphData.edges.forEach(edge => {
        if (!edge._polyline) {
            console.warn(`Edge ${edge.source}-${edge.target} has no polyline reference`);
            return;
        }
        if (edge.is_metro) return;  // Don't color metro
        
        const edgeKey = `${edge.source}-${edge.target}`;
        const congestion = predMap[edgeKey];
        const baseline = baselineMap[edgeKey];
        const isClosed = state.closedRoads.has(edgeKey);
        
        let color = '#3498db';  // Default blue
        let weight = 2;
        let dashArray = null;
        
        if (isClosed) {
            // Blocked road - PINK dashed
            color = '#ff69b4';
            weight = 5;
            dashArray = '12, 6';
        } else if (congestion !== undefined) {
            // Color based on absolute congestion level (5 levels)
            if (congestion >= p80) {
                // Very High - Dark Red
                color = '#c0392b';
                weight = 4;
            } else if (congestion >= p60) {
                // High - Orange
                color = '#e67e22';
                weight = 3;
            } else if (congestion >= p40) {
                // Medium - Yellow
                color = '#f1c40f';
                weight = 2.5;
            } else if (congestion >= p20) {
                // Low - Light Green
                color = '#2ecc71';
                weight = 2;
            } else {
                // Very Low - Dark Green
                color = '#1e8449';
                weight = 2;
            }
            
            // Track changes from baseline
            if (hasClosures && baseline !== undefined) {
                const diff = congestion - baseline;
                if (diff > 0.01) increasedCount++;
                else if (diff < -0.01) decreasedCount++;
            }
        }
        
        edge._polyline.setStyle({ color: color, weight: weight, dashArray: dashArray });
        visualizedCount++;
    });
    
    console.log(`✅ Visualization complete: ${visualizedCount} roads colored`);
    if (hasClosures) {
        console.log(`📊 Traffic redistribution: ${increasedCount} roads got busier, ${decreasedCount} roads got less busy`);
    } else {
        console.log(`📊 Showing current traffic levels (no road closures)`);
    }
}

// ============================================================
// ROAD CLOSURE
// ============================================================

function toggleRoadClosure(edge) {
    const roadId = `${edge.source}-${edge.target}`;
    
    if (state.closedRoads.has(roadId)) {
        state.closedRoads.delete(roadId);
        edge._polyline.setStyle({
            color: CONFIG.COLORS.road,
            weight: 2,
            dashArray: null
        });
        showToast(`Opened road: ${roadId}`, 'info');
    } else {
        state.closedRoads.add(roadId);
        edge._polyline.setStyle({
            color: '#ff69b4',  // Pink
            weight: 4,
            dashArray: '10, 6'
        });
        showToast(`Closed road: ${roadId}`, 'info');
    }
    
    updateClosedRoadsList();
}

function updateClosedRoadsList() {
    const container = document.getElementById('closed-roads-list');
    
    if (state.closedRoads.size === 0) {
        container.innerHTML = '<p class="empty-message">No roads closed</p>';
        return;
    }
    
    container.innerHTML = '';
    state.closedRoads.forEach(roadId => {
        const item = document.createElement('div');
        item.className = 'closed-road-item';
        item.innerHTML = `
            <span>${roadId}</span>
            <button onclick="removeClosedRoad('${roadId}')">×</button>
        `;
        container.appendChild(item);
    });
}

function removeClosedRoad(roadId) {
    state.closedRoads.delete(roadId);
    
    // Update visual
    const edge = state.graphData.edges.find(e => 
        `${e.source}-${e.target}` === roadId
    );
    if (edge && edge._polyline) {
        edge._polyline.setStyle({
            color: CONFIG.COLORS.road,
            dashArray: null
        });
    }
    
    updateClosedRoadsList();
}

function clearClosures() {
    state.closedRoads.forEach(roadId => removeClosedRoad(roadId));
    state.closedRoads.clear();
    updateClosedRoadsList();
    showToast('All road closures cleared', 'info');
}

function randomCloseRoad() {
    if (!state.graphData || !state.graphData.edges) {
        showToast('No graph data available', 'error');
        return;
    }
    
    // Get the count from input field
    const countInput = document.getElementById('random-road-count');
    let count = parseInt(countInput.value) || 1;
    
    // Filter out already closed roads
    const openRoads = state.graphData.edges.filter(edge => {
        const roadId = `${edge.source}-${edge.target}`;
        return !state.closedRoads.has(roadId);
    });
    
    if (openRoads.length === 0) {
        showToast('All roads are already closed', 'warning');
        return;
    }
    
    // Adjust count if it exceeds available roads
    count = Math.min(count, openRoads.length);
    
    // Shuffle and pick 'count' random roads
    const shuffled = openRoads.sort(() => 0.5 - Math.random());
    const selectedRoads = shuffled.slice(0, count);
    
    let blockedCount = 0;
    selectedRoads.forEach(randomEdge => {
        const roadId = `${randomEdge.source}-${randomEdge.target}`;
        
        // Close it
        state.closedRoads.add(roadId);
        
        if (randomEdge._polyline) {
            randomEdge._polyline.setStyle({
                color: '#ff69b4',
                weight: 4,
                dashArray: '10, 6'
            });
        }
        blockedCount++;
    });
    
    updateClosedRoadsList();
    showToast(`🚧 Randomly blocked ${blockedCount} road${blockedCount > 1 ? 's' : ''}`, 'warning');
}

function randomOpenRoad() {
    if (state.closedRoads.size === 0) {
        showToast('No roads are closed', 'info');
        return;
    }
    
    // Get the count from input field
    const countInput = document.getElementById('random-road-count');
    let count = parseInt(countInput.value) || 1;
    
    // Pick random closed roads
    const closedArray = Array.from(state.closedRoads);
    
    // Adjust count if it exceeds closed roads
    count = Math.min(count, closedArray.length);
    
    // Shuffle and select 'count' roads
    const shuffled = closedArray.sort(() => 0.5 - Math.random());
    const selectedRoads = shuffled.slice(0, count);
    
    let openedCount = 0;
    selectedRoads.forEach(randomRoadId => {
        // Open it
        state.closedRoads.delete(randomRoadId);
        
        // Find and update visual
        const edge = state.graphData.edges.find(e => 
            `${e.source}-${e.target}` === randomRoadId
        );
        
        if (edge && edge._polyline) {
            edge._polyline.setStyle({
                color: '#3498db',
                weight: 2,
                dashArray: null
            });
        }
        openedCount++;
    });
    
    updateClosedRoadsList();
    showToast(`✅ Randomly opened ${openedCount} road${openedCount > 1 ? 's' : ''}`, 'success');
}

// ============================================================
// UI UPDATES
// ============================================================

function updateStatusUI(data) {
    const badge = document.getElementById('status-badge');
    const text = document.getElementById('status-text');
    const deviceText = document.getElementById('device-text');
    
    if (data.status === 'online') {
        badge.className = 'status-badge online';
        text.textContent = 'Online';
        
        document.getElementById('model-status').textContent = 
            data.model_loaded ? '✓ Loaded' : '✗ Not loaded';
        document.getElementById('model-status').className = 
            'value ' + (data.model_loaded ? 'success' : 'error');
            
        document.getElementById('graph-status').textContent = 
            data.graph_loaded ? '✓ Loaded' : '✗ Not loaded';
        document.getElementById('graph-status').className = 
            'value ' + (data.graph_loaded ? 'success' : 'error');
            
        document.getElementById('node-count').textContent = data.nodes || '--';
        document.getElementById('edge-count').textContent = data.edges || '--';
        
        deviceText.textContent = data.device?.toUpperCase() || 'CPU';
    } else {
        badge.className = 'status-badge offline';
        text.textContent = 'Offline';
    }
}

function updateStatsUI(stats, baselineStats = null) {
    if (!stats) return;
    
    const toPercent = (val) => {
        if (val === undefined || val === null) return null;
        const num = Number(val);
        if (Number.isNaN(num)) return null;
        return num > 1 ? num : num * 100;
    };
    
    const formatValue = (val) => {
        const pct = toPercent(val);
        if (pct === null) return '--';
        return `${pct.toFixed(1)}%`;
    };
    
    const formatDelta = (current, baseline) => {
        if (!baselineStats) return '';
        const currentPct = toPercent(current);
        const baselinePct = toPercent(baseline);
        if (currentPct === null || baselinePct === null) return '';
        const delta = currentPct - baselinePct;
        if (Math.abs(delta) < 0.1) {
            return `<span class="stat-delta neutral">No change</span>`;
        }
        const direction = delta > 0 ? 'positive' : 'negative';
        const arrow = delta > 0 ? '▲' : '▼';
        return `<span class="stat-delta ${direction}">${arrow} ${Math.abs(delta).toFixed(1)} pp</span>`;
    };
    
    const setStatValue = (elementId, currentValue, baselineValue) => {
        const el = document.getElementById(elementId);
        if (!el) return;
        const deltaHtml = baselineValue !== undefined && baselineValue !== null
            ? formatDelta(currentValue, baselineValue)
            : '';
        el.innerHTML = `${formatValue(currentValue)}${deltaHtml}`;
    };
    
    setStatValue('stat-mean', stats.mean_congestion, baselineStats?.mean_congestion);
    setStatValue('stat-max', stats.max_congestion, baselineStats?.max_congestion);
    setStatValue('stat-road', stats.road_mean, baselineStats?.road_mean);
    setStatValue('stat-metro', stats.metro_mean, baselineStats?.metro_mean);
}

function showNodeInfo(node) {
    const panel = document.getElementById('info-panel');
    const title = document.getElementById('info-title');
    const content = document.getElementById('info-content');
    
    title.textContent = `Node ${node.id}`;
    content.innerHTML = `
        <p><span class="label">Zone:</span> <span class="value">${node.zone}</span></p>
        <p><span class="label">Population:</span> <span class="value">${node.population.toLocaleString()}</span></p>
        <p><span class="label">Amenity:</span> <span class="value">${node.amenity || 'None'}</span></p>
        <p><span class="label">Metro Station:</span> <span class="value">${node.is_metro ? 'Yes' : 'No'}</span></p>
        <p><span class="label">Position:</span> <span class="value">(${node.x.toFixed(2)}, ${node.y.toFixed(2)})</span></p>
    `;
    
    panel.classList.add('visible');
}

function resetSimulation() {
    // Clear all closures
    state.closedRoads.clear();
    state.predictions = null;
    state.baselinePredictions = null;
    state.baselineStats = null;
    
    // Hide View Analysis button
    const analysisBtn = document.getElementById('view-analysis-btn');
    if (analysisBtn) analysisBtn.style.display = 'none';
    
    // Re-render graph to remove highlighting
    if (state.graphData) {
        renderGraph();
    }
    
    // Reset stats
    document.getElementById('stat-mean').textContent = '--';
    document.getElementById('stat-max').textContent = '--';
    document.getElementById('stat-road').textContent = '--';
    document.getElementById('stat-metro').textContent = '--';
    
    // Update closed roads list
    updateClosedRoadsList();
    
    showToast('Simulation reset', 'success');
}

// ============================================================
// UTILITIES
// ============================================================

function showLoading(message = 'Loading...') {
    document.getElementById('loading-text').textContent = message;
    document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'check-circle' : 
                 type === 'error' ? 'exclamation-circle' : 'info-circle';
    
    toast.innerHTML = `<i class="fas fa-${icon}"></i> ${message}`;
    container.appendChild(toast);
    
    setTimeout(() => toast.remove(), 3000);
}

// Make removeClosedRoad available globally for onclick
window.removeClosedRoad = removeClosedRoad;

// ============================================================
// CTM (CELL TRANSMISSION MODEL) FUNCTIONS
// ============================================================

async function initCTM() {
    console.log('🚀 initCTM called');
    showLoading('Initializing CTM... (this may take 10-30 seconds for large graphs)');
    
    try {
        console.log('📡 Fetching CTM initialize endpoint...');
        const startTime = Date.now();
        
        const response = await fetch(`${CONFIG.API_BASE}/ctm/initialize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cell_length_km: 0.5,
                time_step_hours: 1.0/60.0,
                initial_density_ratio: 0.3,
                demand_generation_rate: 100.0
            })
        });
        
        const data = await response.json();
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        
        if (data.status === 'initialized') {
            const cellCount = data.total_cells.toLocaleString();
            showToast(`CTM initialized in ${elapsed}s: ${cellCount} cells created`, 'success');
            console.log(`✅ CTM initialization took ${elapsed}s`);
            updateCTMStats(data.stats);
            
            // Initial visualization
            await updateCTMVisualization();
        } else {
            showToast('Failed to initialize CTM', 'error');
        }
    } catch (error) {
        console.error('CTM init error:', error);
        showToast('Error initializing CTM: ' + error.message, 'error');
    }
    
    hideLoading();
}

async function stepCTM(numSteps = 1) {
    console.log(`🚀 stepCTM called with ${numSteps} steps`);
    showLoading(`Running ${numSteps} simulation step(s)...`);
    
    try {
        console.log('📡 Fetching CTM step endpoint...');
        const response = await fetch(`${CONFIG.API_BASE}/ctm/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steps: numSteps })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showToast(`Completed ${data.steps_completed} steps`, 'success');
            updateCTMStats(data.stats);
            
            // Update visualization with CTM congestion
            await updateCTMVisualization();
        } else {
            showToast('CTM step failed', 'error');
        }
    } catch (error) {
        console.error('CTM step error:', error);
        showToast('Error running CTM step', 'error');
    }
    
    hideLoading();
}

async function updateCTMVisualization() {
    try {
        console.log('🎨 Fetching CTM edge congestion data...');
        const response = await fetch(`${CONFIG.API_BASE}/ctm/edge-congestion`);
        const data = await response.json();
        
        if (!data.edges || data.edges.length === 0) {
            console.warn('⚠️ No CTM edge data received');
            showToast('No CTM data available', 'warning');
            return;
        }
        
        console.log(`📊 Received CTM data for ${data.edges.length} edges`);
        let updatedCount = 0;
        
        if (data.edges && state.layers.roads) {
            // Update road colors based on CTM congestion
            state.layers.roads.eachLayer((layer) => {
                const edgeData = data.edges.find(e => 
                    e.source == layer.feature.source && 
                    e.target == layer.feature.target
                );
                
                if (edgeData) {
                    const color = getCongestionColor(edgeData.congestion);
                    layer.setStyle({ 
                        color: color, 
                        weight: 4,
                        opacity: 0.9
                    });
                    updatedCount++;
                }
            });
            console.log(`✅ Updated ${updatedCount} roads with CTM congestion colors`);
        } else {
            console.warn('⚠️ Missing road layers or CTM data');
        }
    } catch (error) {
        console.error('❌ CTM visualization error:', error);
        showToast('Failed to update CTM visualization', 'error');
    }
}

function getCongestionColor(congestionLevel) {
    // congestionLevel is 0.0 to 1.0 (0% to 100%)
    if (congestionLevel >= 0.8) return '#c0392b';  // Very high - dark red
    if (congestionLevel >= 0.6) return '#e67e22';  // High - orange
    if (congestionLevel >= 0.4) return '#f1c40f';  // Medium - yellow
    if (congestionLevel >= 0.2) return '#2ecc71';  // Low - green
    return '#1e8449';  // Very low - dark green
}

function updateCTMStats(stats) {
    if (!stats) {
        console.warn('No stats provided to updateCTMStats');
        return;
    }
    
    console.log('📊 Updating CTM stats:', stats);
    
    // Update CTM status panel
    const ctmTime = document.getElementById('ctm-time');
    const ctmVehicles = document.getElementById('ctm-vehicles');
    const ctmDensity = document.getElementById('ctm-density');
    
    if (ctmTime) ctmTime.textContent = `${stats.simulation_time?.toFixed(1) || 0} min`;
    if (ctmVehicles) ctmVehicles.textContent = (stats.total_vehicles || 0).toLocaleString();
    if (ctmDensity) ctmDensity.textContent = `${((stats.average_congestion || 0) * 100).toFixed(1)}%`;
    
    // Update main stats panel
    const avgCongestion = stats.average_congestion || stats.mean_congestion || 0;
    const maxCongestion = stats.max_congestion || 0;
    const roadAvg = stats.road_avg_congestion || stats.road_mean || 0;
    const metroAvg = stats.metro_avg_congestion || stats.metro_mean || 0;
    
    document.getElementById('stat-mean').textContent = `${(avgCongestion * 100).toFixed(1)}%`;
    document.getElementById('stat-max').textContent = `${(maxCongestion * 100).toFixed(1)}%`;
    document.getElementById('stat-road').textContent = `${(roadAvg * 100).toFixed(1)}%`;
    document.getElementById('stat-metro').textContent = `${(metroAvg * 100).toFixed(1)}%`;
    
    console.log('✅ Stats updated successfully');
}

async function resetCTM() {
    showLoading('Resetting CTM...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/ctm/reset`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.status === 'reset') {
            showToast('CTM reset successfully', 'success');
            
            // Reset stats display
            document.getElementById('ctm-time').textContent = '0 min';
            document.getElementById('ctm-vehicles').textContent = '0';
            document.getElementById('ctm-density').textContent = '0%';
            
            // Reset visualization
            if (state.graphData) {
                renderGraph();
            }
        } else {
            showToast('Failed to reset CTM', 'error');
        }
    } catch (error) {
        console.error('CTM reset error:', error);
        showToast('Error resetting CTM', 'error');
    }
    
    hideLoading();
}

// CTM road closure - replaces the old method
async function addClosedRoad(source, target) {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/ctm/close-road`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source, target, key: 0 })
        });
        
        const data = await response.json();
        
        if (data.status === 'closed') {
            state.closedRoads.add(`${source}-${target}`);
            updateClosedRoadsList();
            updateCTMStats(data.stats);
            await updateCTMVisualization();
            showToast(`Road ${source} → ${target} closed`, 'info');
        }
    } catch (error) {
        console.error('CTM close road error:', error);
        showToast('Error closing road', 'error');
    }
}

// ============================================================
// EXPOSE FUNCTIONS TO WINDOW FOR DEBUGGING
// ============================================================
window.initCTM = initCTM;
window.stepCTM = stepCTM;
window.resetCTM = resetCTM;
window.updateCTMVisualization = updateCTMVisualization;

// Log when script loads
console.log('✅ CTM functions loaded:', {
    initCTM: typeof initCTM,
    stepCTM: typeof stepCTM,
    resetCTM: typeof resetCTM
});
