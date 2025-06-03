"""
Logging and progress tracking utilities for Image Search System
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger
from tqdm import tqdm
import sys

@dataclass
class ProcessCheckpoint:
    """Checkpoint data structure for process resumption"""
    process_name: str
    total_items: int
    completed_items: int
    completed_ids: List[str]
    start_time: float
    last_update: float
    metadata: Dict[str, Any]
    
    def save(self, checkpoint_file: str):
        """Save checkpoint to file"""
        checkpoint_data = asdict(self)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    @classmethod
    def load(cls, checkpoint_file: str) -> Optional['ProcessCheckpoint']:
        """Load checkpoint from file"""
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None
    
    def get_progress(self) -> float:
        """Get completion progress as percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100
    
    def get_eta(self) -> Optional[float]:
        """Estimate time remaining in seconds"""
        if self.completed_items == 0:
            return None
        
        elapsed = time.time() - self.start_time
        rate = self.completed_items / elapsed
        remaining_items = self.total_items - self.completed_items
        
        if rate > 0:
            return remaining_items / rate
        return None

class ProgressTracker:
    """Enhanced progress tracker with checkpointing and resume capability"""
    
    def __init__(self, 
                 process_name: str,
                 total_items: int,
                 checkpoint_file: str,
                 description: str = "",
                 auto_save_interval: int = 10):
        
        self.process_name = process_name
        self.checkpoint_file = checkpoint_file
        self.auto_save_interval = auto_save_interval
        self._save_counter = 0
        
        # Try to load existing checkpoint
        self.checkpoint = ProcessCheckpoint.load(checkpoint_file)
        
        if self.checkpoint and self.checkpoint.total_items == total_items:
            # Resume from checkpoint
            logger.info(f"📂 Resuming {process_name} from checkpoint: "
                       f"{self.checkpoint.completed_items}/{total_items} completed "
                       f"({self.checkpoint.get_progress():.1f}%)")
            
            self.pbar = tqdm(
                total=total_items,
                initial=self.checkpoint.completed_items,
                desc=f"🔄 {description or process_name}",
                unit="items"
            )
        else:
            # Start fresh
            logger.info(f"🚀 Starting {process_name}: {total_items} items to process")
            
            self.checkpoint = ProcessCheckpoint(
                process_name=process_name,
                total_items=total_items,
                completed_items=0,
                completed_ids=[],
                start_time=time.time(),
                last_update=time.time(),
                metadata={}
            )
            
            self.pbar = tqdm(
                total=total_items,
                desc=f"⚡ {description or process_name}",
                unit="items"
            )
    
    def update(self, item_id: str, metadata: Dict[str, Any] = None):
        """Update progress and checkpoint"""
        if item_id not in self.checkpoint.completed_ids:
            self.checkpoint.completed_ids.append(item_id)
            self.checkpoint.completed_items = len(self.checkpoint.completed_ids)
            self.checkpoint.last_update = time.time()
            
            if metadata:
                self.checkpoint.metadata.update(metadata)
            
            self.pbar.update(1)
            
            # Auto-save checkpoint periodically
            self._save_counter += 1
            if self._save_counter >= self.auto_save_interval:
                self.save_checkpoint()
                self._save_counter = 0
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        self.checkpoint.save(self.checkpoint_file)
    
    def is_completed(self, item_id: str) -> bool:
        """Check if item is already completed"""
        return item_id in self.checkpoint.completed_ids
    
    def get_completed_ids(self) -> List[str]:
        """Get list of completed item IDs"""
        return self.checkpoint.completed_ids.copy()
    
    def set_description(self, desc: str):
        """Update progress bar description"""
        self.pbar.set_description(desc)
    
    def set_postfix(self, **kwargs):
        """Update progress bar postfix"""
        # Add ETA if available
        eta = self.checkpoint.get_eta()
        if eta:
            kwargs['ETA'] = f"{eta/60:.1f}m"
        
        self.pbar.set_postfix(**kwargs)
    
    def close(self):
        """Close progress bar and save final checkpoint"""
        self.save_checkpoint()
        self.pbar.close()
        
        if self.checkpoint.completed_items >= self.checkpoint.total_items:
            logger.success(f"✅ {self.process_name} completed successfully!")
        else:
            logger.warning(f"⚠️  {self.process_name} incomplete: "
                          f"{self.checkpoint.completed_items}/{self.checkpoint.total_items}")

class ProcessLogger:
    """Enhanced logger for the image search system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )
        
        # File handler for detailed logs
        log_file = self.log_dir / f"image_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days"
        )
        
        # Error file handler
        error_file = self.log_dir / "errors.log"
        logger.add(
            error_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="5 MB",
            retention="30 days"
        )
        
        self.logger = logger
    
    def log_process_start(self, process_name: str, details: Dict[str, Any] = None):
        """Log process start with details"""
        msg = f"🚀 Starting process: {process_name}"
        if details:
            msg += f" | Details: {details}"
        self.logger.info(msg)
    
    def log_process_complete(self, process_name: str, duration: float, stats: Dict[str, Any] = None):
        """Log process completion with stats"""
        msg = f"✅ Completed process: {process_name} | Duration: {duration:.2f}s"
        if stats:
            msg += f" | Stats: {stats}"
        self.logger.success(msg)
    
    def log_process_error(self, process_name: str, error: Exception, context: Dict[str, Any] = None):
        """Log process error with context"""
        msg = f"❌ Error in process: {process_name} | Error: {str(error)}"
        if context:
            msg += f" | Context: {context}"
        self.logger.error(msg)
    
    def log_checkpoint_save(self, process_name: str, progress: float):
        """Log checkpoint save"""
        self.logger.debug(f"💾 Checkpoint saved for {process_name}: {progress:.1f}% complete")
    
    def log_system_info(self, info: Dict[str, Any]):
        """Log system information"""
        self.logger.info(f"🖥️  System Info: {info}")

# Global logger instance
process_logger = ProcessLogger()

def create_progress_tracker(process_name: str, 
                           total_items: int, 
                           checkpoint_file: str,
                           description: str = "") -> ProgressTracker:
    """Factory function to create progress tracker"""
    return ProgressTracker(
        process_name=process_name,
        total_items=total_items,
        checkpoint_file=checkpoint_file,
        description=description
    )

def log_system_requirements():
    """Log system requirements and status"""
    import torch
    import platform
    import psutil
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        system_info["cuda_device"] = torch.cuda.get_device_name(0)
    
    process_logger.log_system_info(system_info) 