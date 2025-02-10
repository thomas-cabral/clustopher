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
            
            mapReloadTrigger++;
            
        } catch (error) {
            console.error('Error loading cluster:', error);
        }
    }
    
    function handleClusterCreated() {
        mapReloadTrigger += 1;
    }

    function handleClusterLoaded() {
        mapReloadTrigger += 1;
    }
</script>

<div class="layout">
    <aside class="sidebar">
        <ClusterSelector
            apiBaseUrl={API_BASE_URL}
            on:selectCluster={loadCluster}
            on:clusterCreated={handleClusterCreated}
            on:clusterLoaded={handleClusterLoaded}
        />
    </aside>
    
    <main class="main">
        {#if MAPBOX_TOKEN}
            <ClusterMap
                mapboxToken={MAPBOX_TOKEN}
                apiBaseUrl={API_BASE_URL}
                width="100%"
                height="100%"
                reloadTrigger={mapReloadTrigger}
            />
        {/if}
    </main>
</div>

<style>
    .layout {
        display: grid;
        grid-template-columns: 400px 1fr;
        height: 100vh;
        width: 100vw;
        overflow: hidden;
    }

    .sidebar {
        background: #f5f5f5;
        overflow-y: auto;
        border-right: 1px solid #ddd;
        padding: 1rem;
    }

    .main {
        position: relative;
        height: 100%;
        width: 100%;
    }

    /* Remove any margin/padding from body */
    :global(body) {
        margin: 0;
        padding: 0;
        overflow: hidden;
    }

    :global(#app) {
        height: 100vh;
    }
</style>