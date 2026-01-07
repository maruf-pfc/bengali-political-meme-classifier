const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const resultContainer = document.getElementById('resultContainer');
const resultBadge = document.getElementById('resultBadge');
const loadingSpinner = document.getElementById('loadingSpinner');
const uploadPrompt = document.getElementById('uploadPrompt');

// Drag and Drop Events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    dropZone.classList.add('dragover');
}

function unhighlight(e) {
    dropZone.classList.remove('dragover');
}

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

// Click to Upload
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            uploadFile(file);
            showPreview(file);
        } else {
            alert('Please upload an image file (JPG, PNG).');
        }
    }
}

function showPreview(file) {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function() {
        imagePreview.src = reader.result;
        previewContainer.style.display = 'block';
        uploadPrompt.style.display = 'none';
        resultContainer.style.display = 'none';
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Show loading
    loadingSpinner.style.display = 'block';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();
        showResult(data.prediction);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during classification.');
    } finally {
        loadingSpinner.style.display = 'none';
    }
}

function showResult(prediction) {
    resultBadge.textContent = prediction;
    resultBadge.className = 'badge'; // Reset classes
    
    if (prediction === 'Political') {
        resultBadge.classList.add('political');
    } else {
        resultBadge.classList.add('non-political');
    }
    
    resultContainer.style.display = 'block';
}
