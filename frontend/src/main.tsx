import React from 'react'
import ReactDOM from 'react-dom/client'
import './config/axios'  // Import axios configuration
import './index.css'  // Import beautiful CSS styling
import App from './App'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
) 