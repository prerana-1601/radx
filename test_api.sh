#!/bin/bash
# test_api.sh - Script to test the RadX-CV API

API_URL="http://localhost:8000"

echo "Testing RadX-CV Medical AI API..."

# Test health check
echo "1. Testing health check..."
curl -s "$API_URL/api/health" | python3 -m json.tool

echo -e "\n\n2. Testing image upload..."
# Test image upload (you need to have a test image)
# Replace 'test_image.jpg' with an actual X-ray image path
if [ -f "data/chest_xray/test/NORMAL/IM-0033-0001-0001.jpeg" ]; then
    echo "Uploading test image..."
    RESPONSE=$(curl -s -X POST "$API_URL/api/analyze-xray" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@data/chest_xray/test/NORMAL/IM-0033-0001-0001.jpeg" \
        -F "patient_name=Test Patient" \
        -F "patient_id=TEST001")
    
    echo $RESPONSE | python3 -m json.tool
    
    # Extract analysis ID
    ANALYSIS_ID=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['analysis_id'])")
    
    echo -e "\n\n3. Checking analysis status..."
    sleep 2  # Wait a bit for processing
    
    # Check status multiple times
    for i in {1..10}; do
        echo "Check $i:"
        STATUS_RESPONSE=$(curl -s "$API_URL/api/analysis/$ANALYSIS_ID")
        echo $STATUS_RESPONSE | python3 -m json.tool
        
        STATUS=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
        
        if [ "$STATUS" = "completed" ]; then
            echo -e "\n\n4. Getting final report..."
            curl -s "$API_URL/api/analysis/$ANALYSIS_ID/report" | python3 -m json.tool
            
            echo -e "\n\n5. Downloading GradCAM image..."
            curl -s "$API_URL/api/analysis/$ANALYSIS_ID/gradcam" --output "test_gradcam_$ANALYSIS_ID.jpg"
            echo "GradCAM saved as test_gradcam_$ANALYSIS_ID.jpg"
            break
        elif [ "$STATUS" = "failed" ]; then
            echo "Analysis failed!"
            break
        else
            echo "Status: $STATUS, waiting..."
            sleep 5
        fi
    done
    
    echo -e "\n\n6. Testing list analyses..."
    curl -s "$API_URL/api/analyses" | python3 -m json.tool
    
else
    echo "No test image found at data/chest_xray/test/NORMAL/IM-0033-0001-0001.jpeg"
    echo "Please provide a test image path or create a sample image."
fi

echo -e "\n\nAPI testing complete!"