const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const previewImage = document.getElementById('previewImage');
const predictionResult = document.getElementById('predictionResult');
const confidenceScore = document.getElementById('confidenceScore');

// File upload handling
uploadArea.addEventListener('click', () => imageInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#764ba2';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        analyzeBtn.disabled = false;
        resultsSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Image analysis
analyzeBtn.addEventListener('click', async () => {
    if (!imageInput.files[0]) return;
    
    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    
    try {
        const response = await fetch('/detect_brain_tumor', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        alert('Analysis failed. Please try again.');
    }
    
    analyzeBtn.textContent = 'Analyze Image';
    analyzeBtn.disabled = false;
});

function displayResults(result) {
    const tumorDetected = result.tumor_detected;
    const confidence = (result.confidence * 100).toFixed(2);
    const detections = result.detections || [];
    
    // Update the preview image with annotated version
    if (result.annotated_image) {
        previewImage.src = result.annotated_image;
    }
    
    predictionResult.innerHTML = `
        <div class="result-status ${tumorDetected ? 'tumor-detected' : 'no-tumor'}">
            <strong>${result.message}</strong>
            <p>Total detections: ${result.total_tumors}</p>
        </div>
    `;
    
    confidenceScore.innerHTML = `
        <div class="confidence-section">
            <div class="confidence-score">
                <p>Max Confidence: ${confidence}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                </div>
            </div>
            
            ${detections.length > 0 ? createDetectionsList(detections) : ''}
        </div>
    `;
}

function createDetectionsList(detections) {
    let detectionsHTML = '<div class="detections-list"><h4>Detection Details:</h4>';
    
    detections.forEach((detection, index) => {
        const confidence = (detection.confidence * 100).toFixed(2);
        detectionsHTML += `
            <div class="detection-item">
                <div class="detection-header">
                    <strong>Detection ${index + 1}: ${detection.tumor_type}</strong>
                    <span class="confidence-badge">${confidence}%</span>
                </div>
                <div class="detection-details">
                    <p>Location: (${detection.bbox.x1}, ${detection.bbox.y1}) to (${detection.bbox.x2}, ${detection.bbox.y2})</p>
                    <p>Dimensions: ${detection.dimensions.width} × ${detection.dimensions.height} pixels</p>
                    <p>Area: ${detection.dimensions.area} px²</p>
                    <p>Center: (${detection.center.x}, ${detection.center.y})</p>
                </div>
            </div>
        `;
    });
    
    detectionsHTML += '</div>';
    return detectionsHTML;
}