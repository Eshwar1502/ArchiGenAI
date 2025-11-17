const dropdowns = ['Country', 'cloudStack', 'Welldefined', 'involvesML', 'sourceType', 'dataIngestion', 'etlTool'];
const checkmarks = ['check1', 'check2', 'check3', 'check4', 'check5', 'check6', 'check7'];
const goButton = document.getElementById('goButton');
const progressFill = document.getElementById('progressFill');
const matrixContainer = document.getElementById('matrixContainer');
const jsonContainer = document.getElementById('jsonContainer');
const header = document.getElementById('header');
const matrixBody = document.getElementById('matrixBody');
const jsonCode = document.getElementById('jsonCode');

// Update progress bar based on completed dropdowns
function updateProgress() {
    let completed = 0;
    dropdowns.forEach(id => {
        if (document.getElementById(id).value) {
            completed++;
        }
    });

    const percentage = (completed / dropdowns.length) * 100;
    progressFill.style.width = percentage + '%';

    // Enable button only when all fields are filled
    goButton.disabled = completed !== dropdowns.length;
}

// Add event listeners to all dropdowns
dropdowns.forEach((id, index) => {
    document.getElementById(id).addEventListener('change', function () {
        const checkmark = document.getElementById(checkmarks[index]);
        if (this.value) {
            checkmark.classList.add('show');
        } else {
            checkmark.classList.remove('show');
        }
        updateProgress();
    });
});

// Populate matrix table with data
function populateMatrix(data) {
    matrixBody.innerHTML = '';
    
    if (!data || data.length === 0) {
        console.error('No data to populate matrix');
        alert('No recommendations received from the model');
        return;
    }
    
    data.forEach((row, index) => {
        const tr = document.createElement('tr');
        tr.dataset.rowIndex = index;
        
        // Backend structure: [rank, score, cloud, sourcetype, mode, datalake, dataingestion, tools, workflow, involvesml, welldefined]
        // Desired order: Rank, Cloud, Source Type, Mode, Data Ingestion, Workflow Orchestration, Data Transformation (tools), Data Lake/Warehouse
        
        // Reorder columns to match your requirements
        const reorderedRow = [
            row[0],  // Rank
            row[2],  // Cloud
            row[3],  // Source Type
            row[4],  // Mode
            row[6],  // Data Ingestion
            row[8],  // Workflow Orchestration
            row[7],  // Data Transformation (tools)
            row[5],  // Data Lake/Warehouse
            // Optionally add remaining fields if needed:
            // row[1],  // Score
            // row[9],  // Involves ML
            // row[10]  // Well Defined
        ];
        
        // Add cells in the new order
        reorderedRow.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        
        // Add click handler for row selection - pass original row data
        tr.addEventListener('click', function() {
            handleRowSelection(this, row);
        });
        
        matrixBody.appendChild(tr);
    });
}

// Populate matrix table with data
// function populateMatrix(data) {
//     matrixBody.innerHTML = '';
    
//     if (!data || data.length === 0) {
//         console.error('No data to populate matrix');
//         alert('No recommendations received from the model');
//         return;
//     }
    
//     data.forEach((row, index) => {
//         const tr = document.createElement('tr');
//         tr.dataset.rowIndex = index;
        
//         // Backend structure: [rank, score, cloud, sourcetype, mode, datalake, dataingestion, tools, workflow, involvesml, welldefined]
//         // Desired order: Rank, Cloud, Source Type, Mode, Data Ingestion, Workflow Orchestration, Data Transformation (tools), Data Lake/Warehouse, then remaining
        
//         // Reorder columns to match your requirements + add remaining columns
//         const reorderedRow = [
//             row[0],  // Rank
//             row[2],  // Cloud
//             row[3],  // Source Type
//             row[4],  // Mode
//             row[6],  // Data Ingestion
//             row[8],  // Workflow Orchestration
//             row[7],  // Data Transformation (tools)
//             row[5],  // Data Lake/Warehouse
//             row[1],  // Score (remaining)
//             row[9],  // Involves ML (remaining)
//             row[10]  // Well Defined (remaining)
//         ];
        
//         // Add cells in the new order
//         reorderedRow.forEach(cell => {
//             const td = document.createElement('td');
//             td.textContent = cell;
//             tr.appendChild(td);
//         });
        
//         // Add click handler for row selection - pass original row data
//         tr.addEventListener('click', function() {
//             handleRowSelection(this, row);
//         });
        
//         matrixBody.appendChild(tr);
//     });
// }
// function parseMarkdown(text) {
//     // Convert **bold** to <strong>bold</strong>
//     text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
//     // Convert *italic* to <em>italic</em>
//     text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
//     // Convert # Headers
//     text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
//     text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
//     text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
//     // Convert line breaks
//     text = text.replace(/\n\n/g, '</p><p>');
//     text = text.replace(/\n/g, '<br>');
    
//     // Convert bullet lists
//     text = text.replace(/^\- (.*$)/gim, '<li>$1</li>');
//     text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
//     // Convert numbered lists
//     text = text.replace(/^\d+\.\s+(.*$)/gim, '<li>$1</li>');
    
//     // Wrap in paragraph if not already wrapped
//     if (!text.startsWith('<h') && !text.startsWith('<ul')) {
//         text = '<p>' + text + '</p>';
//     }
    
//     return text;
// }


// // Handle row selection and JSON display
// function handleRowSelection(rowElement, rowData) {
//     // Remove previous selection
//     const previousSelected = matrixBody.querySelector('tr.selected');
//     if (previousSelected) {
//         previousSelected.classList.remove('selected');
//     }
    
//     // Add selection to clicked row
//     rowElement.classList.add('selected');
    
//     // Create JSON object matching your backend structure
//     // rowData: [rank, score, cloud, sourcetype, mode, datalake, dataingestion, tools, workflow, involvesml, welldefined]
//     const jsonData = {
//         rank: rowData[0],
//         cloud: rowData[2],
//         source_type: rowData[3],
//         mode: rowData[4],
//         data_ingestion: rowData[6],
//         workflow_orchestration: rowData[8],
//         data_transformation: rowData[7],
//         datalake_warehouse: rowData[5],
//         score: rowData[1],
//         involves_ml: rowData[9],
//         well_defined: rowData[10]
//     };
    
//     // Display JSON
// // Show loading state while fetching from Gemini
//     jsonCode.textContent = '⏳ Generating architecture details...\n\nPlease wait while we create a comprehensive architecture explanation for your selected configuration.';
//     jsonContainer.classList.add('show');
    
//     // Scroll to JSON container
//     setTimeout(() => {
//         jsonContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
//     }, 100);
    
//     console.log('Selected architecture:', jsonData);
    
//     // Call Gemini to get detailed architecture explanation
//     fetch('http://localhost:8000/gemini-conversation', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify(jsonData)
//     })
//     .then(response => {
//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }
//         return response.json();
//     })
//     .then(data => {
//         console.log('Gemini API response:', data);
        
//         // Display the Gemini-generated architecture explanation
//         if (data.response) {
//             // Replace the loading message with actual response
//             jsonCode.textContent = data.response;
//             console.log('Architecture Details:', data.response);
//         } else {
//             jsonCode.textContent = 'No architecture details received from the server.';
//         }
//     })
//     .catch(error => {
//         console.error('Gemini API error:', error);
//         jsonCode.textContent = `❌ Error generating architecture details:\n\n${error.message}\n\nPlease try again or check the console for more details.`;
//     });
// }

// Helper function to convert basic Markdown to HTML
function parseMarkdown(text) {
    if (!text) return '';
    
    // Convert **bold** to <strong>bold</strong> (handle multi-line and special chars)
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Convert *italic* to <em>italic</em> (only single asterisks not part of **)
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Convert #### Headers
    text = text.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Convert numbered lists (1. item)
    text = text.replace(/^\d+\.\s+(.*)$/gim, '<li>$1</li>');
    
    // Convert bullet lists (- item or * item)
    text = text.replace(/^[\-\*]\s+(.*)$/gim, '<li>$1</li>');
    
    // Wrap consecutive <li> tags in <ul>
    text = text.replace(/(<li>.*?<\/li>\n?)+/gs, '<ul>$&</ul>');
    
    // Convert double line breaks to paragraphs
    text = text.split('\n\n').map(para => {
        if (para.trim() && !para.startsWith('<h') && !para.startsWith('<ul') && !para.startsWith('<li')) {
            return '<p>' + para.replace(/\n/g, '<br>') + '</p>';
        }
        return para;
    }).join('\n');
    
    return text;
}

// Handle row selection and JSON display
function handleRowSelection(rowElement, rowData) {
    // Remove previous selection
    const previousSelected = matrixBody.querySelector('tr.selected');
    if (previousSelected) {
        previousSelected.classList.remove('selected');
    }
    
    // Add selection to clicked row
    rowElement.classList.add('selected');
    
    // Create JSON object matching your backend structure
    // rowData: [rank, score, cloud, sourcetype, mode, datalake, dataingestion, tools, workflow, involvesml, welldefined]
    const jsonData = {
        rank: rowData[0],
        cloud: rowData[2],
        source_type: rowData[3],
        mode: rowData[4],
        data_ingestion: rowData[6],
        workflow_orchestration: rowData[8],
        data_transformation: rowData[7],
        datalake_warehouse: rowData[5],
        score: rowData[1],
        involves_ml: rowData[9],
        well_defined: rowData[10]
    };
    
    // Show loading state while fetching from Gemini - USE innerHTML
    jsonCode.innerHTML = '<p style="text-align: center; color: #4CAF50;"><strong>⏳ Generating architecture details...</strong></p><p>Please wait while we create a comprehensive architecture explanation for your selected configuration.</p>';
    jsonContainer.classList.add('show');
    
    // Scroll to JSON container
    setTimeout(() => {
        jsonContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
    
    console.log('Selected architecture:', jsonData);
    
    // Call Gemini to get detailed architecture explanation
    fetch('http://localhost:8000/gemini-conversation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(jsonData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Gemini API response:', data);
        
        // Display the Gemini-generated architecture explanation
        if (data.response) {
            // Parse Markdown and render as HTML - USE innerHTML
            const htmlContent = parseMarkdown(data.response);
            jsonCode.innerHTML = htmlContent;
            console.log('Architecture Details:', data.response);
        } else {
            jsonCode.innerHTML = '<p style="color: #ff9800;">No architecture details received from the server.</p>';
        }
    })
    .catch(error => {
        console.error('Gemini API error:', error);
        jsonCode.innerHTML = `<p style="color: #f44336;"><strong>❌ Error generating architecture details:</strong></p><p>${error.message}</p><p>Please try again or check the console for more details.</p>`;
    });
}


// Handle Go button click - ONLY API CALLS
goButton.addEventListener('click', function () {
    const selections = {
        country: document.getElementById('Country').value,
        cloud: document.getElementById('cloudStack').value,
        well_defined_use_case: document.getElementById('Welldefined').value,
        involves_ml: document.getElementById('involvesML').value,
        source_type: document.getElementById('sourceType').value,
        mode: document.getElementById('dataIngestion').value,
        confirmed_tools: document.getElementById('etlTool').value,
    };

    console.log('Submitting selections:', selections);

    goButton.disabled = true;
    goButton.textContent = 'Processing...';

    // Call intake endpoint which returns matrix_data directly
    fetch("http://localhost:8000/intake", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(selections)
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Full API response:", data);
        console.log("Matrix data received:", data.matrix_data);
        console.log("Number of solutions:", data.matrix_data?.length);
        
        // Minimize header
        header.classList.add('minimized');
        
        // Populate matrix with backend data
        if (data.matrix_data && Array.isArray(data.matrix_data) && data.matrix_data.length > 0) {
            populateMatrix(data.matrix_data);
            matrixContainer.classList.add('show');
            
            // Scroll to matrix smoothly
            setTimeout(() => {
                matrixContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        } else {
            console.error("Invalid matrix_data:", data);
            throw new Error("No valid matrix data received from server");
        }

        goButton.disabled = false;
        goButton.textContent = 'Go';
    })
    .catch(error => {
        console.error("Error details:", error);
        alert(`Failed to load recommendations: ${error.message}\n\nTroubleshooting:\n1. Ensure FastAPI server is running on localhost:8000\n2. Check that all model files (encoders.pkl, architecture_predictor.pt) are loaded\n3. Open browser DevTools Console (F12) for detailed error logs`);
        goButton.disabled = false;
        goButton.textContent = 'Go';
    });
});
