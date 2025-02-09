<!-- ClusterMap.svelte -->
<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import mapboxgl from 'mapbox-gl';
    import 'mapbox-gl/dist/mapbox-gl.css';
  
    export let mapboxToken: string;
    export let apiBaseUrl = 'http://localhost:8000';
    export let initialZoom = 10;
    export let center = [-122.4, 37.8];
    export let width = '800px';
    export let height = '700px';
    export let className = '';
  
    let map: mapboxgl.Map;
    let mapContainer: HTMLDivElement;
    let isLoading = false;
  
    async function fetchClusters(bounds: mapboxgl.LngLatBounds, zoom: number) {
      isLoading = true;
      try {
        const params = new URLSearchParams({
          zoom: Math.floor(zoom).toString(),
          north: bounds.getNorth().toString(),
          south: bounds.getSouth().toString(),
          east: bounds.getEast().toString(),
          west: bounds.getWest().toString(),
        });
  
        const response = await fetch(`${apiBaseUrl}/api/clusters?${params}`);
        if (!response.ok) throw new Error('Failed to fetch clusters');
        
        const data = await response.json();
        
        // Debug log
        console.log('Received clusters:', data.features.length, data);
        
        const source = map.getSource('clusters') as mapboxgl.GeoJSONSource;
        if (source) {
          source.setData(data);
        }
      } catch (error) {
        console.error('Error fetching clusters:', error);
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
  
        // Initial fetch
        fetchClusters(map.getBounds() as mapboxgl.LngLatBounds, map.getZoom());
      });
  
      // Click handler for clusters
      map.on('click', 'clusters', (e) => {
        if (!e.features?.[0]) return;
        
        const feature = e.features[0];
        const coordinates = feature.geometry.coordinates.slice() as [number, number];
        
        // Ensure proper zoom level change
        const currentZoom = map.getZoom();
        const targetZoom = Math.min(currentZoom + 2, 16); // Zoom in by 2 levels, max zoom 16
        
        map.flyTo({
          center: coordinates,
          zoom: targetZoom,
          speed: 0.5
        });
      });
  
      // Update clusters when map moves
      map.on('moveend', () => {
        fetchClusters(map.getBounds() as mapboxgl.LngLatBounds, map.getZoom());
      });
  
      // Cursor styling
      map.on('mouseenter', 'clusters', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      
      map.on('mouseleave', 'clusters', () => {
        map.getCanvas().style.cursor = '';
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
  </div>
  
  <style>
    .map-wrapper {
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
  
    .map {
      width: 100%;
      height: 100%;
    }
  
    .loading {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 14px;
      z-index: 1;
    }
  </style>