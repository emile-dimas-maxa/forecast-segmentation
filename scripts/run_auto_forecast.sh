#!/bin/bash

# Automated Forecasting and Comparison Script
# This script runs comprehensive forecasting evaluation automatically

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
OUTPUT_PREFIX="auto_forecast_$(date +%Y%m%d_%H%M%S)"
VERBOSE=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated Forecasting and Comparison Script

REQUIRED OPTIONS:
    --segmented-table TABLE     Table with segmented data
    --train-start YYYY-MM-DD    Training start date
    --train-end YYYY-MM-DD      Training end date  
    --test-start YYYY-MM-DD     Test start date
    --test-end YYYY-MM-DD       Test end date

OPTIONAL:
    --config FILE               Path to configuration file
    --output-prefix PREFIX      Output prefix (default: auto_forecast_TIMESTAMP)
    --verbose                   Enable verbose logging
    --help                      Show this help message

ENVIRONMENT VARIABLES (required):
    SNOWFLAKE_ACCOUNT          Your Snowflake account
    SNOWFLAKE_USER             Your Snowflake username
    SNOWFLAKE_PASSWORD         Your Snowflake password
    SNOWFLAKE_WAREHOUSE        Your Snowflake warehouse
    SNOWFLAKE_DATABASE         Your Snowflake database
    SNOWFLAKE_SCHEMA           Your Snowflake schema
    SNOWFLAKE_ROLE             Your Snowflake role (optional)

EXAMPLES:
    # Basic usage
    $0 --segmented-table my_segmented_data \\
       --train-start 2023-01-01 --train-end 2023-12-31 \\
       --test-start 2024-01-01 --test-end 2024-03-31

    # With custom config and verbose output
    $0 --segmented-table my_segmented_data \\
       --train-start 2023-01-01 --train-end 2023-12-31 \\
       --test-start 2024-01-01 --test-end 2024-03-31 \\
       --config my_config.json --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --segmented-table)
            SEGMENTED_TABLE="$2"
            shift 2
            ;;
        --train-start)
            TRAIN_START="$2"
            shift 2
            ;;
        --train-end)
            TRAIN_END="$2"
            shift 2
            ;;
        --test-start)
            TEST_START="$2"
            shift 2
            ;;
        --test-end)
            TEST_END="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SEGMENTED_TABLE" || -z "$TRAIN_START" || -z "$TRAIN_END" || -z "$TEST_START" || -z "$TEST_END" ]]; then
    print_error "Missing required arguments"
    show_usage
    exit 1
fi

# Validate environment variables
required_env_vars=("SNOWFLAKE_ACCOUNT" "SNOWFLAKE_USER" "SNOWFLAKE_PASSWORD" "SNOWFLAKE_WAREHOUSE" "SNOWFLAKE_DATABASE" "SNOWFLAKE_SCHEMA")
missing_vars=()

for var in "${required_env_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    print_error "Missing required environment variables: ${missing_vars[*]}"
    print_info "Please set the following environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "  export $var=your_value"
    done
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

print_info "🚀 Starting Automated Forecasting and Comparison"
print_info "Project directory: $PROJECT_DIR"
print_info "Segmented table: $SEGMENTED_TABLE"
print_info "Train period: $TRAIN_START to $TRAIN_END"
print_info "Test period: $TEST_START to $TEST_END"
print_info "Output prefix: $OUTPUT_PREFIX"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [[ -f ".venv/bin/activate" ]]; then
    print_info "Activating virtual environment..."
    source .venv/bin/activate
else
    print_warning "Virtual environment not found at .venv/bin/activate"
fi

# Build command
CMD="python scripts/auto_forecast_compare.py"
CMD="$CMD --segmented-table '$SEGMENTED_TABLE'"
CMD="$CMD --train-start '$TRAIN_START'"
CMD="$CMD --train-end '$TRAIN_END'"
CMD="$CMD --test-start '$TEST_START'"
CMD="$CMD --test-end '$TEST_END'"
CMD="$CMD --output-prefix '$OUTPUT_PREFIX'"

if [[ -n "$CONFIG_FILE" ]]; then
    CMD="$CMD --config '$CONFIG_FILE'"
fi

if [[ "$VERBOSE" == "true" ]]; then
    CMD="$CMD --verbose"
fi

print_info "Running command: $CMD"

# Run the automated comparison
if eval "$CMD"; then
    print_success "🎉 Automated forecasting comparison completed successfully!"
    print_info "Check the 'reports/' directory for detailed results"
else
    print_error "❌ Automated forecasting comparison failed"
    exit 1
fi
