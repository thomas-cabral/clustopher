#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create output directories
mkdir -p profiles
mkdir -p benchmark_results

echo -e "${GREEN}Running Clustering Benchmarks${NC}"
echo "================================="

# Run specific profiling test
run_specific_profile() {
    echo -e "${BLUE}Running profile for $1 points at zoom level $2${NC}"
    go test -run=^$ -bench="BenchmarkClustering${1}_" -benchmem ./cluster/ -count=3 -benchtime=1s | tee benchmark_results/benchmark_${1}_z${2}.txt
}

# Run all benchmarks
run_all_benchmarks() {
    echo -e "${YELLOW}Running all benchmarks...${NC}"
    go test -run=^$ -bench=. -benchmem ./cluster/ | tee benchmark_results/all_benchmarks.txt
}

# Run profile tests
run_profile_test() {
    echo -e "${BLUE}Running profile comparison test...${NC}"
    go test -run=TestProfileClustering ./cluster/ -v | tee benchmark_results/profile_test.txt
}

# Run CPU profiling
run_cpu_profile() {
    echo -e "${YELLOW}Running CPU profile for $1 points at zoom level $2...${NC}"
    go test -run=^$ -bench="BenchmarkClustering${1}_" -cpuprofile=profiles/cpu_${1}_z${2}.prof ./cluster/
    echo -e "${GREEN}To analyze: go tool pprof profiles/cpu_${1}_z${2}.prof${NC}"
}

# Run memory profiling
run_mem_profile() {
    echo -e "${YELLOW}Running memory profile for $1 points at zoom level $2...${NC}"
    go test -run=^$ -bench="BenchmarkClustering${1}_" -memprofile=profiles/mem_${1}_z${2}.prof ./cluster/
    echo -e "${GREEN}To analyze: go tool pprof profiles/mem_${1}_z${2}.prof${NC}"
}

# Parse command line arguments
case "$1" in
    benchmark)
        run_all_benchmarks
        ;;
    profile)
        run_profile_test
        ;;
    cpu)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Usage: $0 cpu [Small|Medium|Large] [Low|Mid|High]${NC}"
            exit 1
        fi
        run_cpu_profile $2 $3
        ;;
    mem)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Usage: $0 mem [Small|Medium|Large] [Low|Mid|High]${NC}"
            exit 1
        fi
        run_mem_profile $2 $3
        ;;
    specific)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Usage: $0 specific [Small|Medium|Large] [Low|Mid|High]${NC}"
            exit 1
        fi
        run_specific_profile $2 $3
        ;;
    all)
        # Run everything
        run_all_benchmarks
        run_profile_test
        
        # Run CPU profiles for different dataset sizes and zoom levels
        for size in "Small" "Medium" "Large"; do
            for zoom in "Low" "Mid" "High"; do
                run_cpu_profile $size $zoom
            done
        done
        
        # Run memory profiles for different dataset sizes and zoom levels
        for size in "Small" "Medium" "Large"; do
            for zoom in "Low" "Mid" "High"; do
                run_mem_profile $size $zoom
            done
        done
        ;;
    *)
        echo -e "${RED}Usage: $0 [benchmark|profile|cpu|mem|specific|all]${NC}"
        echo "  benchmark: Run all benchmarks"
        echo "  profile: Run profile comparison test"
        echo "  cpu: Run CPU profiling (requires size and zoom params)"
        echo "  mem: Run memory profiling (requires size and zoom params)"
        echo "  specific: Run specific benchmark (requires size and zoom params)"
        echo "  all: Run everything"
        echo ""
        echo "  Size options: Small, Medium, Large"
        echo "  Zoom options: Low, Mid, High"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}" 