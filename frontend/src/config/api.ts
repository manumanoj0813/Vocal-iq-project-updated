import axios from 'axios';

// Create axios instance
const api = axios.create({
  baseURL: '/api', // This will be proxied to http://127.0.0.1:8000 by Vite
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to add auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

export default api; 