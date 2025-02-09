<!-- routes/map/+page.svelte -->
<script lang="ts">
    import ClusterMap from '../../components/ClusterMap.svelte';
    
    const MAPBOX_TOKEN = 'pk.eyJ1IjoidGNhYnJhbCIsImEiOiJjbTZ3eHh6b3IwZnBiMmxwczA1NGJrNWw3In0.Bm0HoSBdzvnAZ4FuoFDgNA';
  </script>
  
  <div class="page">
    <h1>Cluster Map</h1>
    
    <div class="map-container">
      <ClusterMap 
        mapboxToken={MAPBOX_TOKEN}
        apiBaseUrl="http://localhost:8000"
        width="100%"
        height="500px"
        className="custom-map"
        initialZoom={10}
        center={[-122.4, 37.8]}
      />
    </div>
  </div>
  
  <style>
    .page {
      padding: 2rem;
    }
  
    h1 {
      margin-bottom: 1.5rem;
    }
  
    .map-container {
      max-width: 1200px;
      margin: 0 auto;
    }
  </style>