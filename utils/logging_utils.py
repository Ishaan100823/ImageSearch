"""
Enhanced Multi-Model Logging System for Image Search Platform
Features: Model-specific logging, Resume capability, Progress tracking, Performance metrics
"""
import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ModelCheckpoint:
    """Enhanced checkpoint for multi-model resume capability"""
    completed_ids: Set[str]
    total_count: int
    start_time: float
    model_name: str
    stage_name: str
    metadata: Dict[str, Any]
    
    def save(self, file_path: str):
        """Save checkpoint with model context"""
        data = {
            'completed_ids': list(self.completed_ids),
            'total_count': self.total_count,
            'start_time': self.start_time,
            'model_name': self.model_name,
            'stage_name': self.stage_name,
            'metadata': self.metadata,
            'last_updated': time.time()
        }
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> Optional['ModelCheckpoint']:
        """Load checkpoint if exists"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls(
                completed_ids=set(data['completed_ids']),
                total_count=data['total_count'],
                start_time=data['start_time'],
                model_name=data.get('model_name', 'unknown'),
                stage_name=data.get('stage_name', 'unknown'),
                metadata=data.get('metadata', {})
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def is_done(self, item_id: str) -> bool:
        """Check if item already processed - PREVENTS WASTE"""
        return item_id in self.completed_ids
    
    def mark_done(self, item_id: str):
        """Mark item as completed"""
        self.completed_ids.add(item_id)
    
    def progress(self) -> float:
        """Get progress percentage"""
        return len(self.completed_ids) / self.total_count * 100 if self.total_count > 0 else 0
    
    def eta_seconds(self) -> float:
        """Estimate time to completion"""
        if len(self.completed_ids) == 0:
            return 0
        
        elapsed = time.time() - self.start_time
        rate = len(self.completed_ids) / elapsed
        remaining = self.total_count - len(self.completed_ids)
        return remaining / rate if rate > 0 else 0


class ModelProgress:
    """Enhanced progress tracker with model context and ETA"""
    
    def __init__(self, stage_name: str, total_items: int, checkpoint_file: str, model_name: str = "general"):
        self.stage_name = stage_name
        self.model_name = model_name
        self.checkpoint_file = checkpoint_file
        
        # Load or create checkpoint
        self.checkpoint = ModelCheckpoint.load(checkpoint_file)
        if not self.checkpoint:
            self.checkpoint = ModelCheckpoint(set(), total_items, time.time(), model_name, stage_name, {})
            print(f"🚀 [{model_name}] Starting {stage_name}")
        else:
            completed = len(self.checkpoint.completed_ids)
            progress_pct = self.checkpoint.progress()
            eta = self.checkpoint.eta_seconds()
            print(f"📂 [{model_name}] Resuming {stage_name}: {completed}/{total_items} done ({progress_pct:.1f}%, ETA: {eta:.0f}s)")
        
        # Progress bar with model context
        self.pbar = tqdm(
            total=total_items,
            initial=len(self.checkpoint.completed_ids),
            desc=f"[{model_name}] {stage_name}",
            unit="items",
            ncols=100
        )
    
    def skip_if_done(self, item_id: str) -> bool:
        """Check and skip if already done - CORE RESUME FEATURE"""
        return self.checkpoint.is_done(item_id)
    
    def mark_complete(self, item_id: str, metadata: Dict[str, Any] = None):
        """Mark item complete and update progress"""
        if not self.checkpoint.is_done(item_id):
            self.checkpoint.mark_done(item_id)
            if metadata:
                self.checkpoint.metadata[item_id] = metadata
            
            # Update progress bar with ETA
            self.pbar.update(1)
            eta = self.checkpoint.eta_seconds()
            self.pbar.set_postfix({'ETA': f'{eta:.0f}s'})
    
    def save(self):
        """Save progress"""
        self.checkpoint.save(self.checkpoint_file)
    
    def finish(self):
        """Complete the stage"""
        self.save()
        self.pbar.close()
        elapsed = time.time() - self.checkpoint.start_time
        rate = len(self.checkpoint.completed_ids) / elapsed if elapsed > 0 else 0
        print(f"✅ [{self.model_name}] {self.stage_name} completed in {elapsed:.1f}s ({rate:.1f} items/sec)")


class MultiModelLogger:
    """Enhanced logging with model context and file outputs"""
    
    def __init__(self, model_name: str = "general", log_dir: str = "logs"):
        self.model_name = model_name
        self.start_time = time.time()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup both console and file logging"""
        # Create model-specific logger
        self.file_logger = logging.getLogger(f"model_{self.model_name}")
        self.file_logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if self.file_logger.handlers:
            return
        
        # File handler for model-specific logs
        log_file = self.log_dir / f"{self.model_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        
        # General system log
        system_log = self.log_dir / "system.log"
        system_handler = logging.FileHandler(system_log)
        system_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(system_handler)
    
    def info(self, message: str):
        """Info logging with model context"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{self.model_name}] {message}"
        print(formatted_msg)
        self.file_logger.info(message)
    
    def success(self, message: str):
        """Success logging"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{self.model_name}] ✅ {message}"
        print(formatted_msg)
        self.file_logger.info(f"SUCCESS: {message}")
    
    def error(self, message: str):
        """Error logging"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{self.model_name}] ❌ {message}"
        print(formatted_msg, file=sys.stderr)
        self.file_logger.error(message)
    
    def warning(self, message: str):
        """Warning logging"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{self.model_name}] ⚠️ {message}"
        print(formatted_msg)
        self.file_logger.warning(message)
    
    def stage_start(self, stage_name: str):
        """Log stage start with performance tracking"""
        self.info(f"🔄 Starting {stage_name}")
        self._stage_start_time = time.time()
    
    def stage_complete(self, stage_name: str, duration: float = None):
        """Log stage completion with metrics"""
        if duration is None:
            duration = time.time() - getattr(self, '_stage_start_time', time.time())
        
        self.success(f"{stage_name} completed in {duration:.1f}s")
    
    def log_model_info(self, model_config: Dict[str, Any]):
        """Log model configuration information"""
        self.info(f"Model Configuration:")
        self.info(f"  Name: {model_config.get('name', 'Unknown')}")
        self.info(f"  Description: {model_config.get('description', 'No description')}")
        self.info(f"  Categories: {len(model_config.get('categories', []))} categories")
        self.info(f"  Data Directory: {model_config.get('data_dir', 'Unknown')}")
        self.info(f"  Models Directory: {model_config.get('models_dir', 'Unknown')}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.info("Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.3f}")
            else:
                self.info(f"  {key}: {value}")


# Global logger instance (backward compatibility)
logger = MultiModelLogger()


def create_progress(stage_name: str, total_items: int, checkpoint_file: str, model_name: str = "general") -> ModelProgress:
    """Create enhanced progress tracker with model context"""
    return ModelProgress(stage_name, total_items, checkpoint_file, model_name)


def get_model_logger(model_name: str) -> MultiModelLogger:
    """Get model-specific logger"""
    return MultiModelLogger(model_name)


def log_system_requirements():
    """Log comprehensive system information"""
    import torch
    import psutil
    import platform
    
    logger.info("🖥️ System Information:")
    logger.info(f"  Platform: {platform.system()} {platform.release()}")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    memory = psutil.virtual_memory()
    logger.info(f"  RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    disk = psutil.disk_usage('.')
    logger.info(f"  Disk: {disk.free / (1024**3):.1f} GB free of {disk.total / (1024**3):.1f} GB total")


def log_training_summary(model_name: str, metrics: Dict[str, Any]):
    """Log training completion summary"""
    model_logger = get_model_logger(model_name)
    model_logger.success(f"🎉 Training completed for {model_name}")
    model_logger.log_performance_metrics(metrics)


def log_api_startup(models_info: Dict[str, Any]):
    """Log API startup information"""
    logger.info("🚀 API Server Starting")
    logger.info(f"  Available models: {len(models_info)}")
    for model_key, model_data in models_info.items():
        logger.info(f"    • {model_key}: {model_data.get('name', 'Unknown')} ({model_data.get('total_products', 0)} products)")


# Utility functions for common logging patterns
def log_elapsed_time(start_time: float, operation: str, model_name: str = "general"):
    """Log elapsed time for an operation"""
    elapsed = time.time() - start_time
    model_logger = get_model_logger(model_name)
    model_logger.info(f"⏱️ {operation} took {elapsed:.2f}s")


def log_memory_usage(operation: str, model_name: str = "general"):
    """Log current memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        used_gb = (memory.total - memory.available) / (1024**3)
        model_logger = get_model_logger(model_name)
        model_logger.info(f"💾 {operation} - Memory usage: {used_gb:.1f} GB")
    except ImportError:
        pass 