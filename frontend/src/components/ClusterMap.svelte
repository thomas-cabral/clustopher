<!-- ClusterMap.svelte -->
<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import mapboxgl, { type LngLatLike } from 'mapbox-gl';
    import 'mapbox-gl/dist/mapbox-gl.css';
    import { createEventDispatcher } from 'svelte';
  
    const {
        mapboxToken,
        apiBaseUrl = 'http://localhost:8000',
        clusterId = null,
        initialZoom = 10,
        center = [-122.4, 37.8],
        width = '800px',
        height = '700px',
        className = '',
        reloadTrigger = 0
    } = $props<{
        mapboxToken: string;
        apiBaseUrl?: string;
        clusterId?: string | null;
        initialZoom?: number;
        center?: [number, number];
        width?: string;
        height?: string;
        className?: string;
        reloadTrigger?: number;
    }>();
  
    let map: mapboxgl.Map;
    let mapContainer: HTMLDivElement;
    let isLoading = $state(false);
    let metadata: {
        totalPoints: number;
        numClusters: number;
        numSinglePoints: number;
        metricsSummary: Record<string, {
            min: number;
            max: number;
            sum: number;
            average: number;
        }>;
        metadataSummary: Record<string, any>;
    } | null = $state(null);
  
    const dispatch = createEventDispatcher();
  
    $effect(() => {
        if (reloadTrigger && map) {
            const bounds = map.getBounds();
            if (bounds) {
                fetchClusters(bounds, map.getZoom());
            }
        }
    });
  
    async function fetchMetadata(bounds: mapboxgl.LngLatBounds, zoom: number) {
        if (!clusterId) return;
        
        try {
            const params = new URLSearchParams({
                zoom: Math.floor(zoom).toString(),
                north: bounds.getNorth().toString(),
                south: bounds.getSouth().toString(),
                east: bounds.getEast().toString(),
                west: bounds.getWest().toString(),
            });

            const response = await fetch(`${apiBaseUrl}/api/clusters/${clusterId}/metadata?${params}`);
            if (!response.ok) throw new Error('Failed to fetch metadata');
            metadata = await response.json();
        } catch (error) {
            console.error('Error fetching metadata:', error);
        }
    }
  
    async function fetchClusters(bounds: mapboxgl.LngLatBounds, zoom: number) {
        if (!clusterId) return;
        
        isLoading = true;
        try {
            await Promise.all([
                // Existing clusters fetch
                (async () => {
                    const params = new URLSearchParams({
                        zoom: Math.floor(zoom).toString(),
                        north: bounds.getNorth().toString(),
                        south: bounds.getSouth().toString(),
                        east: bounds.getEast().toString(),
                        west: bounds.getWest().toString(),
                    });

                    const response = await fetch(`${apiBaseUrl}/api/clusters/${clusterId}?${params}`);
                    if (!response.ok) throw new Error('Failed to fetch clusters');
                    
                    const data = await response.json();
                    
                    const source = map.getSource('clusters') as mapboxgl.GeoJSONSource;
                    if (source) {
                        source.setData(data);
                    }
                })(),
                // New metadata fetch
                fetchMetadata(bounds, zoom)
            ]);
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            isLoading = false;
        }
    }
  
    onMount(async () => {
      mapboxgl.accessToken = mapboxToken;
  
      map = new mapboxgl.Map({
        container: mapContainer,
        style: 'mapbox://styles/mapbox/dark-v11',
        center: center,
        zoom: initialZoom
      });
  
      map.addControl(new mapboxgl.NavigationControl(), 'top-right');
  
      map.on('load', () => {
        // Add source without Mapbox clustering enabled
        map.addSource('clusters', {
          type: 'geojson',
          data: {
            type: 'FeatureCollection',
            features: []
          }
        });
  
        // Layer for clustered points
        map.addLayer({
          id: 'clusters',
          type: 'circle',
          source: 'clusters',
          filter: ['get', 'cluster'],
          paint: {
            // Size circle radius by point_count
            'circle-radius': [
              'interpolate',
              ['linear'],
              ['get', 'point_count'],
              0, 15,
              100, 25,
              1000, 35,
              10000, 45
            ],
            // Color circles by point_count
            'circle-color': [
              'interpolate',
              ['linear'],
              ['get', 'point_count'],
              0, '#51bbd6',
              100, '#f1f075',
              1000, '#f28cb1'
            ],
            'circle-opacity': 0.8,
            'circle-stroke-width': 2,
            'circle-stroke-color': '#fff',
            'circle-stroke-opacity': 0.5
          }
        });
  
        // Layer for cluster counts
        map.addLayer({
          id: 'cluster-count',
          type: 'symbol',
          source: 'clusters',
          filter: ['get', 'cluster'],
          layout: {
            'text-field': ['get', 'point_count'],
            'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
            'text-size': 12
          },
          paint: {
            'text-color': '#ffffff'
          }
        });
  
        // Layer for unclustered points
        map.addLayer({
          id: 'unclustered-point',
          type: 'circle',
          source: 'clusters',
          filter: ['!', ['get', 'cluster']],
          paint: {
            'circle-color': '#11b4da',
            'circle-radius': 4,
            'circle-stroke-width': 1,
            'circle-stroke-color': '#fff'
          }
        });
  
        // Add popup
        const popup = new mapboxgl.Popup({
          closeButton: false,
          closeOnClick: false
        });
  
        // Show popup on hover for clusters and points
        map.on('mouseenter', 'clusters', (e) => {
          map.getCanvas().style.cursor = 'pointer';
  
          if (!e.features?.[0]) return;
          const feature = e.features[0];
          console.log('Cluster feature:', feature.properties);

          let metrics;
          let metadata;
          try {
            metrics = JSON.parse(feature.properties?.metrics || '{}');
            metadata = JSON.parse(feature.properties?.metadata || '{}');
          } catch (error) {
            console.error('Error parsing feature data:', error);
            metrics = {};
            metadata = {};
          }

          const metricsHtml = Object.entries(metrics)
            .map(([key, value]) => {
              const formattedValue = typeof value === 'number' ? value.toFixed(2) : value;
              const formattedKey = key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');
              return `<p><strong>${formattedKey}:</strong> ${formattedValue}</p>`;
            })
            .join('');

          const metadataHtml = Object.entries(metadata)
            .map(([key, rawValue]) => {
              try {
                // Parse the raw JSON value if it's a string
                const value = typeof rawValue === 'string' ? JSON.parse(rawValue) : rawValue;
                
                let contentHtml = '';
                
                // Handle different types of metadata values
                if (typeof value === 'object') {
                  if (key === 'timeRange') {
                    contentHtml = `
                      <div>From: ${new Date(value.start).toLocaleString()}</div>
                      <div>To: ${new Date(value.end).toLocaleString()}</div>
                    `;
                  } else {
                    // Handle frequency distributions
                    contentHtml = Object.entries(value)
                      .map(([subKey, freq]) => 
                        `<div class="freq-item">
                          <span>${subKey}</span>
                          <span>${typeof freq === 'number' ? freq.toFixed(1) : freq}%</span>
                        </div>`)
                      .join('');
                  }
                } else {
                  // Handle simple values
                  contentHtml = `<div class="freq-item">${value}</div>`;
                }

                return `
                  <div class="metadata-section">
                    <h4>${key.charAt(0).toUpperCase() + key.slice(1)}</h4>
                    <div class="freq-list">
                      ${contentHtml}
                    </div>
                  </div>`;
              } catch (error) {
                console.error(`Error parsing metadata for ${key}:`, error);
                return '';
              }
            })
            .join('');

          const popupContent = `
            <div class="popup-content">
              <h3>Cluster Details</h3>
              <p><strong>Points:</strong> ${feature.properties?.point_count}</p>
              ${metricsHtml}
              ${metadataHtml}
            </div>
          `;

          popup.setLngLat((feature.geometry as { coordinates: number[] }).coordinates.slice() as [number, number])
            .setHTML(popupContent)
            .addTo(map);
        });
  
        // Show popup on hover for individual points
        map.on('mouseenter', 'unclustered-point', (e) => {
          map.getCanvas().style.cursor = 'pointer';
  
          if (!e.features?.[0]) return;
          const feature = e.features[0];

          let metrics;
          try {
            metrics = JSON.parse(feature.properties?.metrics || '{}');
          } catch (error) {
            console.error('Error parsing metrics:', error);
            metrics = {};
          }

          const metricsHtml = Object.entries(metrics)
            .map(([key, value]) => {
              const formattedValue = typeof value === 'number' ? value.toFixed(2) : value;
              const formattedKey = key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');
              return `<p><strong>${formattedKey}:</strong> ${formattedValue}</p>`;
            })
            .join('');

          const popupContent = `
            <div class="popup-content">
              <h3>Point Details</h3>
              ${metricsHtml}
            </div>
          `;

          popup.setLngLat((feature.geometry as { coordinates: number[] }).coordinates.slice() as [number, number])
            .setHTML(popupContent)
            .addTo(map);
        });
  
        map.on('mouseleave', 'clusters', () => {
          map.getCanvas().style.cursor = '';
          popup.remove();
        });
  
        map.on('mouseleave', 'unclustered-point', () => {
          map.getCanvas().style.cursor = '';
          popup.remove();
        });
  
        // Initial fetch
        fetchClusters(map.getBounds() as mapboxgl.LngLatBounds, map.getZoom());
      });
  
      // Click handler for clusters
      map.on('click', 'clusters', (e) => {
        if (!e.features?.[0]) return;
  
        const feature = e.features[0];
        if (feature.geometry.type !== 'Point') return;
  
        const coordinates = (feature.geometry as { coordinates: number[] }).coordinates.slice() as [number, number];
  
        const currentZoom = map.getZoom();
        const targetZoom = Math.min(currentZoom + 2, 16);
  
        map.flyTo({
          center: coordinates,
          zoom: targetZoom,
          speed: 0.5
        });
      });
  
      // Update clusters when map moves
      map.on('moveend', () => {
        const bounds = map.getBounds();
        dispatch('boundsChanged', {
            north: bounds?.getNorth() || 0,
            south: bounds?.getSouth() || 0,
            east: bounds?.getEast() || 0,
            west: bounds?.getWest() || 0,
            zoom: map.getZoom()
        });
        fetchClusters(bounds as mapboxgl.LngLatBounds, map.getZoom());
      });
    });
  
    onDestroy(() => {
      if (map) map.remove();
    });
  </script>
  
  <div class="map-wrapper {className}" style="width: {width}; height: {height}">
    <div bind:this={mapContainer} class="map" />
    {#if isLoading}
      <div class="loading">Loading clusters...</div>
    {/if}
    
    {#if metadata}
        <div class="metadata-panel">
            <h3>View Statistics</h3>
            <div class="stats">
                <div class="stat">
                    <label>Total Points:</label>
                    <span>{metadata.totalPoints?.toLocaleString()}</span>
                </div>
                <div class="stat">
                    <label>Clusters:</label>
                    <span>{metadata.numClusters?.toLocaleString()}</span>
                </div>
                <div class="stat">
                    <label>Individual Points:</label>
                    <span>{metadata.numSinglePoints?.toLocaleString()}</span>
                </div>
            </div>

            {#if Object.keys(metadata?.metricsSummary || {}).length > 0}
                <h4>Metrics Summary</h4>
                <div class="metrics-summary">
                    {#each Object.entries(metadata?.metricsSummary || {}) as [metric, stats]}
                        <div class="metric">
                            <h5>{metric}</h5>
                            <div class="metric-stats">
                                <div>Min: {stats.min.toFixed(2)}</div>
                                <div>Max: {stats.max.toFixed(2)}</div>
                                <div>Avg: {stats.average.toFixed(2)}</div>
                            </div>
                        </div>
                    {/each}
                </div>
            {/if}

            {#if Object.keys(metadata?.metadataSummary || {}).length > 0}
                <h4>Metadata Summary</h4>
                <div class="metadata-summary">
                    {#each Object.entries(metadata.metadataSummary) as [key, value]}
                        <div class="metadata-item">
                            {#if value.time_range}
                                <div class="time-range">
                                    <h5>Time Range</h5>
                                    <div>From: {new Date(value.time_range.earliest).toLocaleString()}</div>
                                    <div>To: {new Date(value.time_range.latest).toLocaleString()}</div>
                                </div>
                            {:else if value.range}
                                <div class="range-section">
                                    <h5>{key.charAt(0).toUpperCase() + key.slice(1)}</h5>
                                    <div class="range-stats">
                                        <div>Min: {value.range.min.toFixed(1)}</div>
                                        <div>Max: {value.range.max.toFixed(1)}</div>
                                        {#if 'average' in value.range}
                                            <div>Avg: {value.range.average.toFixed(1)}</div>
                                        {/if}
                                    </div>
                                </div>
                            {:else if value.distribution}
                                <div class="distribution-section">
                                    <h5>{key.charAt(0).toUpperCase() + key.slice(1)}</h5>
                                    <div class="distribution-list">
                                        {#each Object.entries(value.distribution.values) as [subKey, percentage]}
                                            <div class="distribution-item">
                                                <span>{subKey}</span>
                                                <span>{percentage.toFixed(1)}%</span>
                                            </div>
                                        {/each}
                                    </div>
                                </div>
                            {:else if value.single_value}
                                <div class="single-value">
                                    <h5>{key.charAt(0).toUpperCase() + key.slice(1)}</h5>
                                    <span>{value.single_value}</span>
                                </div>
                            {/if}
                        </div>
                    {/each}
                </div>
            {/if}
        </div>
    {/if}
  </div>
  
  <style>
    /* Apply styles to the popup content directly */
    :global(.mapboxgl-popup-content) {
      padding: 15px;
      border-radius: 6px;
      font-size: 14px;
      line-height: 1.4;
      min-width: 200px;
      background: rgba(255, 255, 255, 0.95);
    }
  
    /* Style the popup close button */
    :global(.mapboxgl-popup-close-button) {
      display: none; /* Hide the close button */
    }
  
    /* Style the popup content wrapper */
    :global(.popup-content) {
      min-width: 150px;
    }
  
    /* Style the strong elements within the popup */
    :global(.popup-content strong) {
      color: #666;
    }
  
    :global(.popup-content h3) {
      margin: 0 0 10px 0;
      font-size: 16px;
      color: #333;
    }
  
    :global(.popup-content p) {
      margin: 5px 0;
    }
  
    .map-wrapper {
      position: relative;
      height: 100%;
      width: 100%;
    }
  
    .map {
      width: 100%;
      height: 100%;
    }
  
    .loading {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 14px;
      z-index: 1;
    }

    .metadata-panel {
        position: absolute;
        bottom: 20px;
        right: 10px;
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-width: 300px;
        max-height: calc(100% - 140px);
        overflow-y: auto;
        z-index: 1;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }

    .metadata-panel h3 {
        margin: 0 0 10px 0;
        font-size: 16px;
        color: #333;
    }

    .metadata-panel h4 {
        margin: 15px 0 8px 0;
        font-size: 14px;
        color: #666;
    }

    .stats {
        display: grid;
        grid-template-columns: 1fr;
        gap: 5px;
    }

    .stat {
        display: flex;
        justify-content: space-between;
        font-size: 14px;
    }

    .metrics-summary {
        display: grid;
        gap: 10px;
    }

    .metric h5 {
        margin: 0 0 5px 0;
        font-size: 13px;
        color: #666;
    }

    .metric-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 5px;
        font-size: 12px;
    }

    .metadata-summary {
        display: grid;
        gap: 5px;
    }

    .metadata-item {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
    }

    .metadata-item label {
        color: #666;
    }

    .time-range {
        font-size: 12px;
        margin: 5px 0;
    }

    .time-range h5 {
        margin: 0 0 5px 0;
        color: #666;
    }

    .distribution-section {
        width: 100%;
        margin: 5px 0;
    }

    .distribution-section h5 {
        margin: 0 0 5px 0;
        color: #666;
        font-size: 13px;
    }

    .distribution-list {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .distribution-item {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        padding: 2px 0;
    }

    .single-value {
        font-size: 12px;
        color: #333;
    }

    /* Add scrollbar styling for the metadata panel */
    .metadata-panel::-webkit-scrollbar {
        width: 8px;
    }

    .metadata-panel::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 4px;
    }

    .metadata-panel::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }

    .metadata-panel::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.3);
    }

    :global(.freq-list) {
        margin-top: 4px;
        font-size: 12px;
    }

    :global(.freq-item) {
        display: flex;
        justify-content: space-between;
        padding: 2px 0;
    }

    :global(.metadata-section) {
        margin-top: 8px;
    }

    :global(.metadata-section h4) {
        margin: 0;
        font-size: 13px;
        color: #666;
    }

    .range-section {
        width: 100%;
        margin: 5px 0;
    }

    .range-section h5 {
        margin: 0 0 5px 0;
        color: #666;
        font-size: 13px;
    }

    .range-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 5px;
        font-size: 12px;
    }
  </style>