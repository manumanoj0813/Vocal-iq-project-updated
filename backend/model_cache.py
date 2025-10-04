import os
import pickle
import logging
from typing import Optional, Any
import time
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCache:
    """Intelligent model caching system for faster loading and memory management"""
    
    def __init__(self, cache_dir: str = "model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache storage
        self._cache = {}
        self._cache_metadata = {}
        self._lock = threading.Lock()
        
        # Cache settings
        self.max_cache_size = 5  # Maximum number of models in memory
        self.cache_ttl = 3600    # Time to live in seconds (1 hour)
        
        logger.info(f"Model cache initialized at: {self.cache_dir}")
    
    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache file path for a model"""
        return self.cache_dir / f"{model_name}.pkl"
    
    def _get_metadata_path(self, model_name: str) -> Path:
        """Get metadata file path for a model"""
        return self.cache_dir / f"{model_name}_metadata.pkl"
    
    def _is_cache_valid(self, model_name: str) -> bool:
        """Check if cached model is still valid"""
        try:
            metadata_path = self._get_metadata_path(model_name)
            if not metadata_path.exists():
                return False
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Check TTL
            if time.time() - metadata.get('timestamp', 0) > self.cache_ttl:
                return False
            
            # Check if cache file exists
            cache_path = self._get_cache_path(model_name)
            if not cache_path.exists():
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache validation failed for {model_name}: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get model from cache if available and valid"""
        try:
            with self._lock:
                # Check if model is in memory
                if model_name in self._cache:
                    logger.info(f"Model {model_name} found in memory cache")
                    return self._cache[model_name]
                
                # Check if model is in disk cache
                if self._is_cache_valid(model_name):
                    cache_path = self._get_cache_path(model_name)
                    with open(cache_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Load into memory cache
                    self._cache[model_name] = model
                    self._cache_metadata[model_name] = {
                        'timestamp': time.time(),
                        'size': os.path.getsize(cache_path)
                    }
                    
                    logger.info(f"Model {model_name} loaded from disk cache")
                    return model
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting model {model_name} from cache: {e}")
            return None
    
    def cache_model(self, model_name: str, model: Any) -> bool:
        """Cache a model for future use"""
        try:
            with self._lock:
                # Check cache size limit
                if len(self._cache) >= self.max_cache_size:
                    self._evict_oldest()
                
                # Store in memory
                self._cache[model_name] = model
                self._cache_metadata[model_name] = {
                    'timestamp': time.time(),
                    'size': 0  # Will be updated when saved to disk
                }
                
                # Save to disk
                cache_path = self._get_cache_path(model_name)
                with open(cache_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Save metadata
                metadata_path = self._get_metadata_path(model_name)
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self._cache_metadata[model_name], f)
                
                logger.info(f"Model {model_name} cached successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error caching model {model_name}: {e}")
            return False
    
    def _evict_oldest(self):
        """Evict the oldest model from memory cache"""
        try:
            if not self._cache_metadata:
                return
            
            # Find oldest model
            oldest_model = min(
                self._cache_metadata.keys(),
                key=lambda k: self._cache_metadata[k]['timestamp']
            )
            
            # Remove from memory
            if oldest_model in self._cache:
                del self._cache[oldest_model]
                del self._cache_metadata[oldest_model]
                logger.info(f"Evicted oldest model: {oldest_model}")
                
        except Exception as e:
            logger.warning(f"Error evicting oldest model: {e}")
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cache for specific model or all models"""
        try:
            with self._lock:
                if model_name:
                    # Clear specific model
                    if model_name in self._cache:
                        del self._cache[model_name]
                        del self._cache_metadata[model_name]
                    
                    # Remove from disk
                    cache_path = self._get_cache_path(model_name)
                    metadata_path = self._get_metadata_path(model_name)
                    
                    if cache_path.exists():
                        cache_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    logger.info(f"Cleared cache for model: {model_name}")
                else:
                    # Clear all models
                    self._cache.clear()
                    self._cache_metadata.clear()
                    
                    # Remove all cache files
                    for file in self.cache_dir.glob("*.pkl"):
                        file.unlink()
                    
                    logger.info("Cleared all model cache")
                    
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            with self._lock:
                total_size = sum(
                    os.path.getsize(self._get_cache_path(name))
                    for name in self._cache.keys()
                    if self._get_cache_path(name).exists()
                )
                
                return {
                    "models_in_memory": len(self._cache),
                    "total_cache_size_bytes": total_size,
                    "cache_directory": str(self.cache_dir),
                    "max_cache_size": self.max_cache_size,
                    "cache_ttl_seconds": self.cache_ttl,
                    "cached_models": list(self._cache.keys())
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def preload_models(self, model_loaders: dict):
        """Preload commonly used models"""
        try:
            logger.info("Starting model preloading...")
            
            for model_name, loader_func in model_loaders.items():
                if self.get_model(model_name) is None:
                    logger.info(f"Preloading model: {model_name}")
                    model = loader_func()
                    if model is not None:
                        self.cache_model(model_name, model)
                        logger.info(f"Successfully preloaded: {model_name}")
                    else:
                        logger.warning(f"Failed to load model: {model_name}")
            
            logger.info("Model preloading completed")
            
        except Exception as e:
            logger.error(f"Error during model preloading: {e}")

# Global cache instance
model_cache = ModelCache()

def get_cached_model(model_name: str, loader_func=None):
    """Get model from cache or load it if not cached"""
    model = model_cache.get_model(model_name)
    
    if model is None and loader_func:
        logger.info(f"Loading model {model_name}...")
        model = loader_func()
        if model is not None:
            model_cache.cache_model(model_name, model)
        else:
            logger.error(f"Failed to load model {model_name}")
    
    return model

def clear_model_cache(model_name: str = None):
    """Clear model cache"""
    model_cache.clear_cache(model_name)

def get_cache_stats():
    """Get cache statistics"""
    return model_cache.get_cache_stats()
