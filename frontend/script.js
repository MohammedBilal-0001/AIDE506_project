document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("csvFile");
  const uploadBtn = document.getElementById("uploadBtn");
  const fileName = document.getElementById("fileName");
  const errorDiv = document.getElementById("error");
  const successDiv = document.getElementById("success");
  const loadingDiv = document.getElementById("loading");
  const resultSection = document.getElementById("resultSection");
  const tableHeader = document.getElementById("tableHeader");
  const tableBody = document.getElementById("tableBody");
  const rowCount = document.getElementById("rowCount");
  let processedData = null; // Store processed data for later access

  fileInput.addEventListener("change", function () {
    if (this.files && this.files[0]) {
      fileName.textContent = this.files[0].name;
      uploadBtn.disabled = false;
      errorDiv.style.display = "none";
      successDiv.style.display = "none";
      resultSection.style.display = "none";
    } else {
      fileName.textContent = "No file selected";
      uploadBtn.disabled = true;
    }
  });

  uploadBtn.addEventListener("click", async function () {
    const file = fileInput.files[0];
    if (!file) {
      showError("Please select a CSV file first.");
      return;
    }

    loadingDiv.style.display = "block";
    uploadBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:5001/process_csv", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Server error");
      }

      const data = await response.json();
      console.log("Server response:", data); // Add this line
      successDiv.textContent = "File processed successfully!";
      successDiv.style.display = "block";
      displayResults(data);
    } catch (error) {
      showError(error.message);
    } finally {
      loadingDiv.style.display = "none";
      uploadBtn.disabled = false;
    }
  });

  function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = "block";
    successDiv.style.display = "none";
    resultSection.style.display = "none";
  }

  function displayResults(data) {
   
    processedData = data;
    // Clear existing content
    rowCount.textContent = `${data.row_count} rows`;
    tableHeader.innerHTML = "";
    tableBody.innerHTML = "";
    
    // Add checkbox column header
    const checkboxHeader = document.createElement("th");
    checkboxHeader.className = "checkbox-column";
    checkboxHeader.innerHTML = '<input type="checkbox" id="selectAll">';
    tableHeader.appendChild(checkboxHeader);

    // Add data headers
    data.columns.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col;
      tableHeader.appendChild(th);
    });

    // Add all rows with checkboxes
    data.sample_data.forEach((row, index) => {
      const tr = document.createElement("tr");

      // Checkbox cell
      const tdCheckbox = document.createElement("td");
      tdCheckbox.className = "checkbox-column";
      tdCheckbox.innerHTML = `<input type="checkbox" data-index="${index}">`;
      tr.appendChild(tdCheckbox);

      // Data cells
      data.columns.forEach((col) => {
        const td = document.createElement("td");
        let value = row[col];
        if (value === null || value === undefined) value = "null";
        if (typeof value === "object") value = JSON.stringify(value);
        td.textContent = value;
        tr.appendChild(td);
      });

      tableBody.appendChild(tr);
    });

    // Add select all functionality
    document
      .getElementById("selectAll")
      .addEventListener("change", function (e) {
        const checkboxes = document.querySelectorAll(
          "#dataTable tbody input[type='checkbox']"
        );
        checkboxes.forEach((checkbox) => (checkbox.checked = e.target.checked));
      });

    // Add predict button
    const predictButton = document.createElement("button");
    predictButton.className = "btn predict-selected-btn";
    predictButton.textContent = "Predict Selected Rows";
    predictButton.onclick = predictSelectedRows;
    resultSection.appendChild(predictButton);

    resultSection.style.display = "block";
  }

 
  async function predictSelectedRows() {
    // Clear previous predictions
    const existingPredictions = document.querySelector(".prediction-results");
    if (existingPredictions) existingPredictions.remove();

    // Validate processedData exists and has sample_data
    if (!processedData || !processedData.sample_data) {
      showError(
        "No data available for prediction. Please upload and process a CSV first."
      );
      return;
    }

    const checkboxes = document.querySelectorAll(
      "#dataTable tbody input[type='checkbox']:checked"
    );

    // Validate checkboxes
    if (checkboxes.length === 0) {
      showError("Please select at least one row to predict");
      return;
    }

    const predictionsDiv = document.createElement("div");
    predictionsDiv.className = "prediction-results";
    predictionsDiv.innerHTML = "<h3>Prediction Results:</h3>";

    try {
      for (const checkbox of checkboxes) {
        const index = parseInt(checkbox.getAttribute("data-index"));

        // Validate index range
        if (
          isNaN(index) ||
          index < 0 ||
          index >= processedData.sample_data.length
        ) {
          showError(`Invalid row index: ${index}`);
          continue;
        }

        const rowData = processedData.sample_data[index];

        // Validate row data
        if (!rowData || typeof rowData !== "object") {
          showError(`Invalid data for row ${index + 1}`);
          continue;
        }

        const response = await fetch("http://localhost:5002/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(rowData),
        });

        if (!response.ok)
          throw new Error(
            `Row ${index + 1}: Prediction failed (${response.status})`
          );

        const prediction = await response.json();

       
        // predictionsDiv.appendChild(predictionItem);
        const predictionItem = document.createElement("div");
            predictionItem.className = "prediction-item";

            // Format probabilities
            const churnProb = (prediction.churn_probability * 100).toFixed(1);
            const nonChurnProb = (prediction.non_churn_probability * 100).toFixed(1);

            // Create and style probability elements
            const churnElement = document.createElement("p");
            churnElement.innerHTML = `Churn Probability: <span class="${churnProb === '100.0' ? 'danger' : ''}">${churnProb}%</span>`;
            
            const nonChurnElement = document.createElement("p");
            nonChurnElement.innerHTML = `Non-Churn Probability: <span class="${nonChurnProb === '100.0' ? 'success' : ''}">${nonChurnProb}%</span>`;

            // Add prediction result
            predictionItem.innerHTML = `<p><strong>Row ${index + 1}:</strong> ${prediction.prediction}</p>`;
            predictionItem.appendChild(churnElement);
            predictionItem.appendChild(nonChurnElement);

            // Add border based on prediction
            predictionItem.style.borderLeft = `4px solid ${prediction.prediction === "Yes" ? "#dc3545" : "#28a745"}`;

            // Add to results container
            predictionsDiv.appendChild(predictionItem);  
      }
    } catch (error) {
      showError(`Prediction error: ${error.message}`);
    } finally {
      // Uncheck checkboxes and clean up
      document
        .querySelectorAll('#dataTable input[type="checkbox"]')
        .forEach((checkbox) => {
          checkbox.checked = false;
        });
      resultSection.appendChild(predictionsDiv);
    }
  }
})
