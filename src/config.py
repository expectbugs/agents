#!/usr/bin/env python3
"""
Configuration Management for Multi-Agent System
Centralized configuration using environment variables with secure defaults
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

class Config:
    """Centralized configuration management with environment variables"""
    
    # Redis Configuration
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')
    REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))
    REDIS_TIMEOUT: int = int(os.getenv('REDIS_TIMEOUT', '30'))
    
    # Model Configuration
    MODEL_PATH: str = os.getenv('MODEL_PATH', '/home/user/models/Hermes-2-Pro-Mistral-10.7B-Q6_K/Hermes-2-Pro-Mistral-10.7B-Q6_K.gguf')
    GPU_LAYERS: int = int(os.getenv('GPU_LAYERS', '-1'))  # -1 = use all GPU layers
    MODEL_CONTEXT_SIZE: int = int(os.getenv('MODEL_CONTEXT_SIZE', '2048'))
    MODEL_TEMPERATURE: float = float(os.getenv('MODEL_TEMPERATURE', '0.7'))
    
    # System Configuration
    WORKSPACE_DIR: Path = Path(os.getenv('WORKSPACE_DIR', '/home/user/agents/workspace'))
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    MAX_TASK_QUEUE_SIZE: int = int(os.getenv('MAX_TASK_QUEUE_SIZE', '100'))
    MAX_ERROR_COUNT: int = int(os.getenv('MAX_ERROR_COUNT', '5'))
    
    # Security Configuration
    MAX_INPUT_LENGTH: int = int(os.getenv('MAX_INPUT_LENGTH', '10000'))
    ALLOWED_COMMANDS: list = os.getenv('ALLOWED_COMMANDS', 'ls,pwd,python --version,pip list').split(',')
    ENABLE_COMMAND_EXECUTION: bool = os.getenv('ENABLE_COMMAND_EXECUTION', 'false').lower() == 'true'
    
    # Performance Configuration
    REDIS_POOL_SIZE: int = int(os.getenv('REDIS_POOL_SIZE', '10'))
    ASYNC_TIMEOUT: int = int(os.getenv('ASYNC_TIMEOUT', '300'))  # 5 minutes
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis connection configuration"""
        config = {
            'url': cls.REDIS_URL,
            'decode_responses': True,
            'retry_on_timeout': True,
            'socket_timeout': cls.REDIS_TIMEOUT,
            'db': cls.REDIS_DB
        }
        
        if cls.REDIS_PASSWORD:
            config['password'] = cls.REDIS_PASSWORD
            
        return config
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get LLM model configuration"""
        return {
            'model_path': cls.MODEL_PATH,
            'n_gpu_layers': cls.GPU_LAYERS,
            'n_ctx': cls.MODEL_CONTEXT_SIZE,
            'temperature': cls.MODEL_TEMPERATURE,
            'verbose': False
        }
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration values"""
        # Ensure workspace directory exists
        cls.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Validate model path if specified
        if cls.MODEL_PATH and not os.path.exists(cls.MODEL_PATH):
            print(f"Warning: Model path {cls.MODEL_PATH} does not exist")
            
        # Validate numeric ranges
        if cls.MAX_TASK_QUEUE_SIZE < 1:
            raise ValueError("MAX_TASK_QUEUE_SIZE must be positive")
            
        if cls.MAX_INPUT_LENGTH < 100:
            raise ValueError("MAX_INPUT_LENGTH must be at least 100")

# Validate configuration on import
Config.validate_config()