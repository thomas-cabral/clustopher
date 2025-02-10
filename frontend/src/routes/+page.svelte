<script>
  import { goto } from '$app/navigation';

  function navigateToMap() {
    goto('/map');
  }
</script>

<main class="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-16 px-4">
  <div class="max-w-3xl mx-auto">
    <div class="bg-white rounded-2xl shadow-xl p-8 md:p-12">
      <h1 class="text-4xl md:text-5xl font-bold text-gray-800 mb-8 text-center">
        Cluster Visualization Project
      </h1>
      
      <div class="space-y-6 text-gray-600 text-lg mb-12">
        <p>
          Welcome to the Cluster Visualization tool. This application helps you visualize 
          and analyze cluster data through an interactive map interface.
        </p>
        <p>
          Explore different cluster configurations and their geographical distribution 
          with our intuitive mapping system.
        </p>
      </div>

      <div class="flex justify-center">
        <button 
          on:click={navigateToMap}
          class="group relative inline-flex items-center justify-center px-8 py-4 text-lg font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors duration-300 overflow-hidden"
        >
          <span class="relative z-10 flex items-center gap-2">
            Open Map View
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              class="h-5 w-5 transform group-hover:translate-x-1 transition-transform duration-200" 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path 
                stroke-linecap="round" 
                stroke-linejoin="round" 
                stroke-width="2" 
                d="M17 8l4 4m0 0l-4 4m4-4H3" 
              />
            </svg>
          </span>
        </button>
      </div>
    </div>

    <div class="mt-8 text-center text-sm text-gray-500">
      <p>
        A powerful tool for visualizing and analyzing geographical cluster data
      </p>
    </div>
  </div>
</main>

<style>
  .container {
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
    text-align: center;
  }

  h1 {
    color: #333;
    margin-bottom: 2rem;
  }

  .description {
    margin-bottom: 2rem;
    line-height: 1.6;
  }

  .map-button {
    background-color: #4CAF50;
    color: white;
    padding: 1rem 2rem;
    border: none;
    border-radius: 4px;
    font-size: 1.1rem;
    cursor: pointer;
    transition: background-color 0.3s;
  }

  .map-button:hover {
    background-color: #45a049;
  }
</style>
