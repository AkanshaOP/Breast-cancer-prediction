
// Constants for feature names (must match the backend model feature names)
const features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness',
    'mean concavity',
    'mean concave points',
    'mean symmetry',
    'mean fractal dimension'
];

async function predict() {
    const data = {};
    let allValid = true;

    // Function to get element ID from feature name
    // e.g. "mean radius" -> "mean_radius"
    const getId = (name) => name.replace(/ /g, '_');

    // Get patient name
    const patientNameInput = document.getElementById('patient_name');
    const patientName = patientNameInput ? patientNameInput.value.trim() : "Patient";
    if (!patientName) {
        alert("Please enter patient name.");
        return;
    }

    // 1. Gather input values
    features.forEach(feature => {
        const id = getId(feature);
        const input = document.getElementById(id);
        if (input && input.value !== "") {
            data[feature] = parseFloat(input.value);
        } else {
            console.warn(`Missing value for ${feature} (id: ${id})`);
            allValid = false;
        }
    });

    if (!allValid) {
        alert("Please fill in all fields.");
        return;
    }

    // 2. Add loading state if you want
    const btn = document.querySelector('.predict-btn');
    const originalText = btn.innerText;
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    try {
        // 3. Send request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // 4. Update UI
        const resultBox = document.getElementById('result');
        const predictionText = document.getElementById('prediction-text');
        const confidenceText = document.getElementById('confidence-text');
        const adviseText = document.getElementById('advice-text');

        if (response.ok) {
            resultBox.classList.remove('hidden');
            resultBox.classList.remove('benign', 'malignant'); // Clear old classes

            if (result.is_malignant) {
                // Malignant
                resultBox.classList.add('malignant');
                predictionText.innerText = `Result for ${patientName}: ${result.prediction}`;
                adviseText.innerText = "Please consult an oncologist immediately for further diagnosis.";
                // Additional UI feedback?
            } else {
                // Benign
                resultBox.classList.add('benign');
                predictionText.innerText = `Result for ${patientName}: ${result.prediction}`;
                adviseText.innerText = "The result appears benign. Regular check-ups are still recommended.";
            }
        } else {
            alert("Error: " + result.error);
        }

    } catch (error) {
        console.error('Error:', error);
        alert("An error occurred while connecting to the server.");
    } finally {
        // Reset button
        btn.innerText = originalText;
        btn.disabled = false;
    }
}
