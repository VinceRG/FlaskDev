// D:\FlaskDev\static\main.js

function showUploadLoader() {
    const fileInput = document.getElementById('file-upload');
    if (fileInput.files.length > 0) {
         const overlay = document.getElementById('loading-overlay');
         overlay.querySelector('#loader-title').textContent = 'Processing Data...';
         overlay.querySelector('#loader-text').textContent = 'Running data pipeline. This may take a moment...';
         overlay.style.display = 'flex';
    }
    // If no files, the form's 'required' attribute will stop submission
}

function showTrainLoader() {
    const overlay = document.getElementById('loading-overlay');
    overlay.querySelector('#loader-title').textContent = 'Training Model...';
    overlay.querySelector('#loader-text').textContent = 'This may take several minutes. Please do not close this window.';
    overlay.style.display = 'flex';
}

// We wrap the event listener in 'DOMContentLoaded' to ensure
// all the HTML elements are loaded before we try to find them.
document.addEventListener('DOMContentLoaded', (event) => {
    
    const fileUploadInput = document.getElementById('file-upload');
    
    // Check if the element actually exists before adding a listener
    if (fileUploadInput) {
        fileUploadInput.addEventListener('change', function(event) {
            const fileList = document.getElementById('file-list');
            const placeholder = document.getElementById('file-list-placeholder');
            
            fileList.innerHTML = ''; // Clear the list
            const files = event.target.files;
            
            if (files.length > 0) {
                placeholder.style.display = 'none';
                for (let i = 0; i < files.length; i++) {
                    const li = document.createElement('li');
                    li.textContent = files[i].name;
                    fileList.appendChild(li);
                }
            } else {
                placeholder.style.display = 'block';
            }
        });
    }
});