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
            dispatch('clusterLoaded');
            loadedClusterId = clusterId;
            
        } catch (err: unknown) {
            error = err instanceof Error ? err.message : 'Unknown error occurred';
            console.error('Error loading cluster:', err);
        } finally {
            loading = false;
        }
    }
    
    // Load clusters on mount
    onMount(loadClusters);
</script>

<div class="cluster-selector">
    <div class="header">
        <h3>Cluster Management</h3>
        <button class="refresh" on:click={loadClusters} disabled={loading}>
            🔄 Refresh
        </button>
    </div>
    
    <div class="new-cluster">
        <h4>Create New Cluster</h4>
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
    
    {#if loading}
        <div class="loading">Loading...</div>
    {:else}
        <h4>Available Clusters</h4>
        <div class="clusters-list">
            {#each clusters as cluster}
                <div class="cluster-item">
                    <div class="cluster-info">
                        <span class="points">{cluster.numPoints.toLocaleString()} points</span>
                        <span class="timestamp">
                            {new Date(cluster.timestamp).toLocaleString()}
                        </span>
                        <span class="size">
                            {(cluster.fileSize / (1024 * 1024)).toFixed(1)} MB
                        </span>
                    </div>
                    <button class="load-button"
                        on:click={() => loadCluster(cluster.id)}
                        disabled={loading || cluster.id === loadedClusterId}
                    >
                        Load
                    </button>
                </div>
            {/each}
            {#if clusters.length === 0}
                <div class="no-clusters">No saved clusters found</div>
            {/if}
        </div>
    {/if}

    {#if loadedClusterInfo}
        <div class="loaded-cluster-info">
            <h4>Loaded Cluster</h4>
            <p><strong>ID:</strong> {loadedClusterInfo.id}</p>
            <p><strong>Points:</strong> {loadedClusterInfo.numPoints.toLocaleString()}</p>
            <p><strong>Timestamp:</strong> {new Date(loadedClusterInfo.timestamp).toLocaleString()}</p>
            <p><strong>File Size:</strong> {(loadedClusterInfo.fileSize / (1024 * 1024)).toFixed(1)} MB</p>
        </div>
    {/if}
</div>

<style>
    .cluster-selector {
        padding: 1rem;
        background: #f5f5f5;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    h4 {
        margin: 1rem 0 0.5rem 0;
        color: #666;
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
    
    .new-cluster {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .clusters-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .cluster-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        background: white;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .cluster-info {
        display: flex;
        gap: 1rem;
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
        background: #4CAF50;
    }
    
    .load-button:hover {
        background: #45a049;
    }

    .loaded-cluster-info {
        margin-top: 1rem;
        padding: 1rem;
        background: white;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    .loaded-cluster-info h4 {
        margin-top: 0;
    }

    .loaded-cluster-info p {
        margin: 0.25rem 0;
    }
</style> 