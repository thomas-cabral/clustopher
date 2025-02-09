<!-- routes/map/+page.svelte -->
<script lang="ts">
    import ClusterMap from '../../components/ClusterMap.svelte';
    import ClusterSelector from '../../components/ClusterSelector.svelte';
    
    const MAPBOX_TOKEN = 'pk.eyJ1IjoidGNhYnJhbCIsImEiOiJjbTZ3eHh6b3IwZnBiMmxwczA1NGJrNWw3In0.Bm0HoSBdzvnAZ4FuoFDgNA';
    const API_BASE_URL = 'http://localhost:8000';
    
    let mapReloadTrigger = 0;
    
    async function handleClusterSelect(event: CustomEvent<string>) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/clusters/load/${event.detail}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to load cluster');
            }
            
            // Increment trigger to force map reload
            mapReloadTrigger += 1;
        } catch (err) {
            console.error('Error loading cluster:', err);
            alert('Failed to load cluster');
        }
    }
    
    function handleClusterCreated() {
        // Increment trigger to force map reload
        mapReloadTrigger += 1;
    }
</script>

<div class="container">
    <ClusterSelector
        apiBaseUrl={API_BASE_URL}
        on:selectCluster={handleClusterSelect}
        on:clusterCreated={handleClusterCreated}
    />
    
    <ClusterMap
        mapboxToken={MAPBOX_TOKEN}
        apiBaseUrl={API_BASE_URL}
        width="100%"
        height="600px"
        reloadTrigger={mapReloadTrigger}
    />
</div>

<style>
    .container {
        padding: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
</style>