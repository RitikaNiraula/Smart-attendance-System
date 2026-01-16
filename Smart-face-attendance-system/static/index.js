let currentStudentId = null;

document.addEventListener('DOMContentLoaded', function () {
    // Add event listeners for buttons
    document.getElementById('saveBtn').addEventListener('click', saveData);
    document.getElementById('updateBtn').addEventListener('click', updateData);
    document.getElementById('deleteBtn').addEventListener('click', deleteData);
    document.getElementById('resetBtn').addEventListener('click', resetData);
    document.getElementById('showAllBtn').addEventListener('click', fetchStudentData);

    // Automatically fetch data when the page loads
    fetchStudentData();
});

// Display success or error messages
function displayMessage(message, type) {
    const messageContainer = document.createElement('div');
    messageContainer.textContent = message;
    messageContainer.className = `message ${type}`;
    document.body.appendChild(messageContainer);
    setTimeout(() => {
        document.body.removeChild(messageContainer);
    }, 3000);
}

// Fetch student data from the server and display it in the table
function fetchStudentData() {
    fetch('http://localhost:3001/student')
        .then(response => response.json())
        .then(data => {
            const studentTable = document.getElementById('studentTable').getElementsByTagName('tbody')[0];
            studentTable.innerHTML = ''; // Clear previous data
            data.forEach(student => {
                const row = studentTable.insertRow();

                // Insert cells for each field
                const cellValues = [
                    student.dep,
                    student.year,
                    student.semester,
                    student.std_id,
                    student.std_name,
                    student.class_group,
                    student.email || 'N/A', // Handle null email
                    student.teacher
                ];

                cellValues.forEach(value => {
                    const cell = row.insertCell();
                    cell.textContent = value;
                });

                // Add a delete button for each student
                const deleteCell = row.insertCell();
                const deleteButton = document.createElement('button');
                deleteButton.textContent = 'Delete';
                deleteButton.classList.add('deleteBtn');
                deleteButton.addEventListener('click', function () {
                    currentStudentId = student.std_id;
                    deleteData();
                });
                deleteCell.appendChild(deleteButton);
            });
        })
        .catch(error => console.error('Error fetching students:', error));
}

// Save student data (with incremented student ID)
function saveData() {
    fetch('http://localhost:3001/latest-student-id')
        .then(response => response.json())
        .then(data => {
            let newStudentId = 1; // Default if no students exist
            if (data.latestStudentId) {
                newStudentId = parseInt(data.latestStudentId) + 1; // Increment by 1
            }

            const studentData = getFormData();
            studentData.studentId = newStudentId;

            fetch('http://localhost:3001/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(studentData)
            })
            .then(response => response.json())
            .then(data => {
                displayMessage(data.message, 'success');
                fetchStudentData(); // Reload table after save
            })
            .catch(error => {
                console.error('Error saving data:', error);
                displayMessage('There was an error saving the data.', 'error');
            });
        })
        .catch(error => {
            console.error('Error fetching latest student ID:', error);
            displayMessage('Error fetching latest student ID.', 'error');
        });
}

// Update student data
function updateData() {
    if (currentStudentId === null) {
        alert('Select a student to update');
        return;
    }

    const studentData = getFormData();
    studentData.studentId = currentStudentId;

    fetch('http://localhost:3001/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(studentData)
    })
    .then(response => response.json())
    .then(data => {
        displayMessage(data.message, 'success');
        fetchStudentData(); // Reload table after update
    })
    .catch(error => {
        console.error('Error updating data:', error);
        displayMessage('There was an error updating the data.', 'error');
    });
}

// Delete student data
function deleteData() {
    if (currentStudentId === null) {
        alert('Select a student to delete');
        return;
    }

    if (confirm('Are you sure you want to delete this student?')) {
        fetch('http://localhost:3001/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ studentId: currentStudentId })
        })
        .then(response => response.json())
        .then(data => {
            displayMessage(data.message, 'success');
            fetchStudentData(); // Reload table after deletion
        })
        .catch(error => {
            console.error('Error deleting data:', error);
            displayMessage('There was an error deleting the data.', 'error');
        });
    }
}

// Reset form fields
function resetData() {
    document.getElementById('studentId').value = '';
    document.getElementById('studentName').value = '';
    document.getElementById('classGroup').value = 'AB';
    document.getElementById('teacherName').value = '';
    document.getElementById('department').value = 'Select Department';
    document.getElementById('year').value = '2024-2025';
    document.getElementById('semester').value = 'Semester-6';
}

// Get form data
function getFormData() {
    return {
        department: document.getElementById('department').value,
        year: document.getElementById('year').value,
        semester: document.getElementById('semester').value,
        studentId: document.getElementById('studentId').value,
        studentName: document.getElementById('studentName').value,
        classGroup: document.getElementById('classGroup').value,
        teacherName: document.getElementById('teacherName').value,
    };
}
