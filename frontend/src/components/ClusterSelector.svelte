<script lang="ts">
    import { createEventDispatcher, onMount } from 'svelte';
    
    const { apiBaseUrl } = $props<{ apiBaseUrl: string }>();
    
    let clusters: Array<{
        id: string;
        numPoints: number;
        timestamp: string;
        fileSize: number;
    }> = $state([]);
    let loading = $state(false);
    let newClusterPoints = $state("300000");
    let error = $state('');
    let loadedClusterInfo: ClusterInfo | null = $state(null);
    let loadedClusterId = $state<string | null>(null);
    
    const dispatch = createEventDispatcher();
    
    interface ClusterInfo {
        id: string;
        numPoints: number;
        timestamp: string;
        fileSize: number;
        metrics?: { [key: string]: number };
    }
    
    async function loadClusters() {
        loading = true;
        error = '';
        try {
            const response = await fetch(`${apiBaseUrl}/api/clusters/list`);
            if (!response.ok) throw new Error('Failed to fetch clusters');
            clusters = await response.json();
        } catch (err: unknown) {
            error = err instanceof Error ? err.message : 'Unknown error occurred';
            console.error('Error loading clusters:', err);
        } finally {
            loading = false;
        }
    }
    
    async function createNewCluster() {
        loading = true;
        error = '';
        try {
            const points = parseInt(newClusterPoints);
            if (isNaN(points) || points <= 0) {
                throw new Error('Invalid number of points');
            }
            
            console.log('Creating new cluster with', points, 'points');
            const response = await fetch(`${apiBaseUrl}/api/clusters`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ numPoints: points }),
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to create cluster');
            }
            
            console.log('Cluster created, waiting before refresh...');
            // Give the server a moment to initialize the new cluster
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            console.log('Refreshing cluster list...');
            await loadClusters();
            dispatch('clusterCreated');
            console.log('Cluster creation complete');
        } catch (err: unknown) {
            error = err instanceof Error ? err.message : 'Unknown error occurred';
            console.error('Error creating cluster:', err);
        } finally {
            loading = false;
        }
    }
    
    async function loadCluster(clusterId: string) {
        loading = true;
        error = '';
        try {
            const response = await fetch(`${apiBaseUrl}/api/clusters/load/${clusterId}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || errorData.message || 'Failed to load cluster');
            }
            
            const responseData: { message: string, clusterInfo: ClusterInfo } = await response.json();
            loadedClusterInfo = responseData.clusterInfo;
            
            // Fetch initial metrics for the whole dataset
            const bounds = {
                north: 90,
                south: -90,
                east: 180,
                west: -180,
                zoom: 0
            };
            
            const metricsResponse = await fetch(
                `${apiBaseUrl}/api/clusters?zoom=0&north=90&south=-90&east=180&west=-180`
            );
            
            if (metricsResponse.ok) {
                const data = await metricsResponse.json();
                if (data.features?.[0]?.properties?.metrics) {
                    loadedClusterInfo.metrics = data.features[0].properties.metrics;
                }
            }
            
            dispatch('clusterLoaded');
            loadedClusterId = clusterId;
            
        } catch (err: unknown) {
            error = err instanceof Error ? err.message : 'Unknown error occurred';
            console.error('Error loading cluster:', err);
        } finally {
            loading = false;
        }
    }
    
    function formatMetricName(name: string): string {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    function formatMetricValue(value: number): string {
        if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        } else if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        } else {
            return value.toFixed(1);
        }
    }
    
    // Load clusters on mount
    onMount(loadClusters);
</script>

<div class="cluster-selector">
    <div class="controls-section">
        <div class="header">
            <h3>Cluster Management</h3>
            <button class="refresh" on:click={loadClusters} disabled={loading}>
                🔄 Refresh
            </button>
        </div>
        
        <div class="new-cluster">
            <input
                type="number"
                bind:value={newClusterPoints}
                placeholder="Number of points"
                min="1"
                disabled={loading}
            />
            <button on:click={createNewCluster} disabled={loading}>
                {loading ? 'Creating...' : 'Create New Cluster'}
            </button>
        </div>
        
        {#if error}
            <div class="error">{error}</div>
        {/if}
    </div>

    <div class="clusters-section">
        <h4>Available Clusters</h4>
        {#if loading}
            <div class="loading">Loading...</div>
        {:else}
            <div class="clusters-list">
                {#each clusters as cluster}
                    <div class="cluster-item {cluster.id === loadedClusterId ? 'active' : ''}">
                        <div class="cluster-info">
                            <div class="cluster-main">
                                <span class="points">{cluster.numPoints.toLocaleString()} points</span>
                                <button class="load-button"
                                    on:click={() => loadCluster(cluster.id)}
                                    disabled={loading || cluster.id === loadedClusterId}
                                >
                                    {cluster.id === loadedClusterId ? 'Loaded' : 'Load'}
                                </button>
                            </div>
                            <div class="cluster-details">
                                <span class="timestamp">{new Date(cluster.timestamp).toLocaleString()}</span>
                                <span class="size">{(cluster.fileSize / (1024 * 1024)).toFixed(1)} MB</span>
                            </div>
                        </div>
                    </div>
                {/each}
                {#if clusters.length === 0}
                    <div class="no-clusters">No saved clusters found</div>
                {/if}
            </div>
        {/if}
    </div>

    {#if loadedClusterInfo && loadedClusterInfo.metrics}
        <div class="loaded-cluster-info">
            <h4>Current Cluster Metrics</h4>
            <div class="metrics-grid">
                {#each Object.entries(loadedClusterInfo.metrics) as [metric, value]}
                    <div class="metric-item">
                        <label>{formatMetricName(metric)}:</label>
                        <span>{formatMetricValue(value)}</span>
                    </div>
                {/each}
            </div>
        </div>
    {/if}
</div>

<style>
    .cluster-selector {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        height: 100%;
    }

    .controls-section {
        padding-bottom: 1rem;
        border-bottom: 1px solid #ddd;
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .new-cluster {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 0.5rem;
        align-items: center;
    }

    .clusters-section {
        flex: 1;
        overflow-y: auto;
        min-height: 0;
    }

    .clusters-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .cluster-item {
        background: white;
        padding: 0.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        transition: background-color 0.2s;
    }

    .cluster-item.active {
        background: #f0f7ff;
        border: 1px solid #4a90e2;
    }

    .cluster-info {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .cluster-main {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .cluster-details {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #666;
    }

    .points {
        font-weight: 500;
        color: #333;
    }

    .loaded-cluster-info {
        padding: 1rem;
        background: white;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-top: auto;
    }

    .loaded-cluster-info h4 {
        margin: 0 0 0.75rem 0;
        color: #333;
        font-size: 1rem;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.5rem;
    }

    .metric-item {
        display: flex;
        flex-direction: column;
        background: #f5f5f5;
        padding: 0.5rem;
        border-radius: 4px;
    }

    .metric-item label {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.25rem;
    }

    .metric-item span {
        font-size: 1.1rem;
        font-weight: 500;
        color: #333;
    }

    .refresh {
        padding: 0.25rem 0.5rem;
        background: transparent;
        border: 1px solid #4a90e2;
        color: #4a90e2;
    }
    
    .refresh:hover {
        background: #4a90e2;
        color: white;
    }
    
    .error {
        color: red;
        margin: 1rem 0;
    }
    
    button {
        padding: 0.5rem 1rem;
        border-radius: 4px;
        border: none;
        background: #4a90e2;
        color: white;
        cursor: pointer;
    }
    
    button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    input {
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    
    .no-clusters {
        text-align: center;
        padding: 1rem;
        color: #666;
        background: #f9f9f9;
        border-radius: 4px;
    }
    
    .load-button {
        padding: 0.25rem 0.75rem;
        font-size: 0.9rem;
        background: #4CAF50;
    }
    
    .load-button:disabled {
        background: #4a90e2;
    }
</style> 