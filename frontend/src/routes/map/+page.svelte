<!-- routes/map/+page.svelte -->
<script lang="ts">
    import ClusterMap from '../../components/ClusterMap.svelte';
    import ClusterSelector from '../../components/ClusterSelector.svelte';
    
    const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
    const API_BASE_URL = 'http://localhost:8000';
    
    let mapReloadTrigger = 0;
    
    async function loadCluster(event: CustomEvent<string>) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/clusters/load/${event.detail}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to load cluster');
            }
            
            // Trigger map refresh
            mapReloadTrigger++;
            
        } catch (error) {
            console.error('Error loading cluster:', error);
            // Handle error (show to user)
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
        on:selectCluster={loadCluster}
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