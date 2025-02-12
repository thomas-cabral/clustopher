<!-- routes/map/+page.svelte -->
<script lang="ts">
    import ClusterMap from '../../components/ClusterMap.svelte';
    import ClusterSelector from '../../components/ClusterSelector.svelte';
    import { page } from '$app/stores';
    import { goto } from '$app/navigation';
    
    const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
    const API_BASE_URL = 'http://localhost:8000';
    
    let mapReloadTrigger = 0;
    let currentBounds = $state({
        north: 90,
        south: -90,
        east: 180,
        west: -180,
        zoom: 0
    });

    // Get the initial cluster ID from URL if present
    let selectedClusterId = $derived($page.url.searchParams.get('cluster'));
    
    function setClusterId(clusterId: string | null) {
        // Update URL with selected cluster ID
        const url = new URL(window.location.href);
        if (clusterId) {
            url.searchParams.set('cluster', clusterId);
        } else {
            url.searchParams.delete('cluster');
        }
        goto(url.toString(), { replaceState: true });
        mapReloadTrigger++;
    }
    
    function handleClusterCreated() {
        mapReloadTrigger += 1;
    }

    function handleClusterLoaded() {
        mapReloadTrigger += 1;
    }

    function handleBoundsChanged(event: CustomEvent) {
        currentBounds = event.detail;
    }
</script>

<div class="layout">
    <aside class="sidebar">
        <ClusterSelector
            apiBaseUrl={API_BASE_URL}
            currentBounds={currentBounds}
            selectedClusterId={selectedClusterId}
            on:setClusterId={event => setClusterId(event.detail)}
            on:clusterCreated={handleClusterCreated}
            on:clusterLoaded={handleClusterLoaded}
        />
    </aside>
    
    <main class="main">
        {#if MAPBOX_TOKEN}
            <ClusterMap
                mapboxToken={MAPBOX_TOKEN}
                apiBaseUrl={API_BASE_URL}
                clusterId={selectedClusterId}
                width="100%"
                height="100%"
                reloadTrigger={mapReloadTrigger}
                on:boundsChanged={handleBoundsChanged}
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