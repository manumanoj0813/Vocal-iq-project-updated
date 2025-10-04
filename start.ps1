# Start the backend server
Start-Process -NoNewWindow powershell -ArgumentList "cd backend; python -m uvicorn main:app --reload"

# Start the frontend development server
Start-Process -NoNewWindow powershell -ArgumentList "cd frontend; npm run dev"

Write-Host "Servers started! Access the application at http://localhost:5173"
Write-Host "Press Ctrl+C to stop all servers"

# Keep the script running
while ($true) { Start-Sleep 1 } 