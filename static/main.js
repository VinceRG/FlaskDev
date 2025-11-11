// ==========================
// LOADING OVERLAY FUNCTIONS
// ==========================
function showUploadLoader() {
    const fileInput = document.getElementById('file-upload');
    if (fileInput.files.length > 0) {
        const overlay = document.getElementById('loading-overlay');
        overlay.querySelector('#loader-title').textContent = 'Processing Data...';
        overlay.querySelector('#loader-text').textContent = 'Running data pipeline. This may take a moment...';
        overlay.style.display = 'flex';
    }
}

function showTrainLoader() {
    const overlay = document.getElementById('loading-overlay');
    overlay.querySelector('#loader-title').textContent = 'Training Model...';
    overlay.querySelector('#loader-text').textContent = 'This may take several minutes. Please do not close this window.';
    overlay.style.display = 'flex';
}


// ===========================================
// MAIN EVENT LISTENER
// ===========================================
document.addEventListener('DOMContentLoaded', () => {

    // --- Setup file list preview (before submission) ---
    const fileUploadInput = document.getElementById('file-upload');
    if (fileUploadInput) {
        fileUploadInput.addEventListener('change', (event) => {
            const fileList = document.getElementById('file-list');
            const placeholder = document.getElementById('file-list-placeholder');
            fileList.innerHTML = '';
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

    // --- Load the list of already-uploaded files ---
    loadUploadedFiles();
});


// ===========================================
// LOAD, DOWNLOAD, AND REMOVE UPLOADED FILES
// (WITH PAGINATION)
// ===========================================

// Store files globally to avoid re-fetching on page change
let allUploadedFiles = [];
const filesPerPage = 10; // Set how many files to show per page

async function loadUploadedFiles() {
    const fileContainer = document.getElementById('uploaded-files-container');
    const paginationContainer = document.getElementById('uploaded-files-pagination');
    if (!fileContainer) return;

    try {
        const res = await fetch('/api/files');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load files');

        allUploadedFiles = data.files; // Store the fetched files

        if (allUploadedFiles.length === 0) {
            fileContainer.innerHTML = '<p class="small-text">No uploaded files found.</p>';
            paginationContainer.innerHTML = ''; // Clear pagination
            return;
        }

        // Setup pagination and display the first page
        setupUploadedFilesPagination();
        displayUploadedFilesPage(1);

    } catch (err) {
        console.error(err);
        fileContainer.innerHTML = '<p class="error-text">Error loading files.</p>';
    }
}

function setupUploadedFilesPagination() {
    const paginationContainer = document.getElementById('uploaded-files-pagination');
    paginationContainer.innerHTML = ''; // Clear old pagination buttons
    
    const totalPages = Math.ceil(allUploadedFiles.length / filesPerPage);

    // Only show pagination if there's more than one page
    if (totalPages <= 1) return;

    for (let i = 1; i <= totalPages; i++) {
        const btn = document.createElement('button');
        btn.textContent = i;
        btn.dataset.page = i;
        // Use the class from your CSS
        // btn.classList.add('pagination-btn'); 

        btn.addEventListener('click', () => {
            displayUploadedFilesPage(i);
        });
        
        paginationContainer.appendChild(btn);
    }
}

function displayUploadedFilesPage(page) {
    const fileContainer = document.getElementById('uploaded-files-container');
    fileContainer.innerHTML = ''; // Clear the list for the new page

    const start = (page - 1) * filesPerPage;
    const end = start + filesPerPage;
    const paginatedFiles = allUploadedFiles.slice(start, end);

    paginatedFiles.forEach(file => {
        const div = document.createElement('div');
        div.classList.add('uploaded-file-item');
        div.innerHTML = `
            <span class="file-name">${file}</span>
            <div class="file-actions">
                <button class="btn-download" onclick="downloadFile('${file}')">⬇️</button>
            </div>
        `;
        fileContainer.appendChild(div);
    });

    // Update active button state
    const paginationContainer = document.getElementById('uploaded-files-pagination');
    const pageButtons = paginationContainer.querySelectorAll('button');
    pageButtons.forEach(btn => btn.classList.remove('active'));
    const activeBtn = paginationContainer.querySelector(`button[data-page="${page}"]`);
    if (activeBtn) activeBtn.classList.add('active');
}

function downloadFile(filename) {
    window.location.href = `/api/download/${encodeURIComponent(filename)}`;
}

