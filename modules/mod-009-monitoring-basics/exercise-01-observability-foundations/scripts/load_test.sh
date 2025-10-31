#!/bin/bash
set -e

echo "=========================================="
echo "Inference Gateway - Load Test"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8000}"
REQUESTS="${REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-10}"

echo "Configuration:"
echo "  Base URL: $BASE_URL"
echo "  Requests: $REQUESTS"
echo "  Concurrency: $CONCURRENCY"
echo ""

# Check if service is ready
echo "Checking if service is ready..."
if ! curl -f -s "$BASE_URL/ready" > /dev/null; then
    echo -e "${RED}✗ Service not ready${NC}"
    echo "Start the service with: docker-compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ Service is ready${NC}"
echo ""

# Create a test image (simple red square)
echo "Creating test image..."
TEST_IMAGE="/tmp/test_image.jpg"
python3 -c "
from PIL import Image
img = Image.new('RGB', (224, 224), color='red')
img.save('$TEST_IMAGE')
"
echo -e "${GREEN}✓ Test image created${NC}"
echo ""

# Function to make a prediction request
make_request() {
    curl -s -X POST "$BASE_URL/predict" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@$TEST_IMAGE" \
        > /dev/null 2>&1
}

# Run load test
echo "Running load test..."
echo "Sending $REQUESTS requests with concurrency $CONCURRENCY..."
echo ""

start_time=$(date +%s)
success=0
failed=0

for ((i=1; i<=REQUESTS; i++)); do
    # Run requests in background for concurrency
    if [ $((i % CONCURRENCY)) -eq 0 ]; then
        wait  # Wait for current batch to complete
    fi

    if make_request; then
        ((success++))
    else
        ((failed++))
    fi &

    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo -ne "\rProgress: $i/$REQUESTS requests sent..."
    fi
done

wait  # Wait for all remaining requests

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo ""
echo "=========================================="
echo "Load Test Results"
echo "=========================================="
echo ""
echo "Total Requests:    $REQUESTS"
echo "Successful:        $success"
echo "Failed:            $failed"
echo "Duration:          ${duration}s"
echo "Requests/sec:      $((REQUESTS / duration))"
echo ""

# Show metrics
echo "=========================================="
echo "Prometheus Metrics Sample"
echo "=========================================="
echo ""
curl -s "$BASE_URL/metrics" | grep -E "http_requests_total|http_request_duration_seconds|model_predictions_total" | head -10
echo ""

echo "View detailed metrics:"
echo "  Prometheus: http://localhost:9090"
echo "  Jaeger:     http://localhost:16686"
echo ""

# Show recent logs
echo "=========================================="
echo "Recent Logs (last 5)"
echo "=========================================="
echo ""
docker-compose logs --tail=5 inference-gateway | grep -E "INFO|ERROR" || echo "No logs available"
echo ""

# Cleanup
rm -f "$TEST_IMAGE"

echo "Load test complete!"
