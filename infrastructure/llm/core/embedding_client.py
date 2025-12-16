"""Ollama embedding client for generating text embeddings.

Provides client for interacting with Ollama's embedding API to generate
semantic embeddings from text using models like embeddinggemma.
"""
from __future__ import annotations

import json
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import requests
from requests.exceptions import Timeout, ConnectionError as RequestsConnectionError

from infrastructure.core.logging_utils import get_logger
from infrastructure.core.exceptions import LLMConnectionError, LLMError
from infrastructure.llm.core.config import LLMConfig

logger = get_logger(__name__)

# Try to import Ollama utilities for health checks and restart
try:
    from infrastructure.llm.utils.ollama import (
        is_ollama_running,
        restart_ollama_server,
        ensure_ollama_ready
    )
    OLLAMA_UTILS_AVAILABLE = True
except ImportError:
    OLLAMA_UTILS_AVAILABLE = False
    logger.warning("Ollama utilities not available - health checks and restart disabled")


class RequestMonitor:
    """Monitor long-running requests with periodic progress updates.
    
    Uses a background thread to log progress and timeout warnings
    during blocking API requests. Automatically logs heartbeats and
    timeout warnings at 50%, 75%, and 90% of timeout elapsed.
    
    The monitor runs in a daemon thread and automatically stops when
    the request completes or fails.
    
    Example:
        >>> monitor = RequestMonitor(timeout=120.0, text_length=50000, heartbeat_interval=10.0)
        >>> monitor.start()
        >>> # ... make blocking request ...
        >>> monitor.stop()
    """
    
    def __init__(
        self,
        timeout: float,
        text_length: int,
        heartbeat_interval: float = 20.0
    ):
        """Initialize request monitor.
        
        Args:
            timeout: Request timeout in seconds. Used to calculate warning thresholds.
            text_length: Length of text being processed (for context in log messages).
            heartbeat_interval: Interval between heartbeat logs in seconds (default: 20.0).
                For very long documents (>50K chars), use 5-10s intervals.
                For medium documents (10K-50K chars), use 10-15s intervals.
        """
        self.timeout = timeout
        self.text_length = text_length
        self.heartbeat_interval = heartbeat_interval
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        self.warning_thresholds = {
            0.5: False,  # 50% of timeout
            0.75: False,  # 75% of timeout
            0.9: False   # 90% of timeout
        }
    
    def start(self) -> None:
        """Start monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            # Thread will exit on next check
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        last_heartbeat_time = 0.0
        
        while self.monitoring:
            elapsed = time.time() - self.start_time
            remaining = self.timeout - elapsed
            
            # Check timeout warning thresholds
            for threshold_pct, warned in self.warning_thresholds.items():
                threshold_time = self.timeout * threshold_pct
                if elapsed >= threshold_time and not warned:
                    self.warning_thresholds[threshold_pct] = True
                    warning_msg = (
                        f"  ⚠ Embedding request at {threshold_pct*100:.0f}% of timeout "
                        f"({elapsed:.1f}s/{self.timeout:.1f}s elapsed, "
                        f"{remaining:.1f}s remaining, text length: {self.text_length:,} chars)"
                    )
                    # Note: RequestMonitor doesn't have access to client instance
                    # So we use print+logger directly here
                    print(warning_msg)
                    logger.warning(warning_msg.strip())
                    sys.stdout.flush()
            
            # Log heartbeat every heartbeat_interval seconds
            if elapsed - last_heartbeat_time >= self.heartbeat_interval:
                heartbeat_msg = (
                    f"  ↻ Still processing embedding request... "
                    f"({elapsed:.1f}s elapsed, {remaining:.1f}s remaining, "
                    f"text length: {self.text_length:,} chars)"
                )
                # Note: RequestMonitor doesn't have access to client instance
                # So we use print+logger directly here
                print(heartbeat_msg)
                logger.info(heartbeat_msg.strip())
                sys.stdout.flush()
                last_heartbeat_time = elapsed
            
            # Check if we've exceeded timeout (shouldn't happen, but safety check)
            if elapsed >= self.timeout:
                logger.error(
                    f"Request monitor detected timeout exceeded "
                    f"({elapsed:.1f}s > {self.timeout:.1f}s)"
                )
                break
            
            # Sleep for a short interval before next check
            time.sleep(1.0)
    
    def get_elapsed(self) -> float:
        """Get elapsed time since monitoring started."""
        return time.time() - self.start_time


class EmbeddingClient:
    """Client for generating embeddings using Ollama embedding models.
    
    Provides methods for:
    - Single embedding generation
    - Batch embedding generation
    - Text chunking for large documents
    - Document-level embedding aggregation
    - Embedding caching
    
    Example:
        >>> client = EmbeddingClient()
        >>> embedding = client.generate_embedding("Machine learning is fascinating")
        >>> embeddings = client.generate_embeddings_batch(["Text 1", "Text 2"])
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        embedding_model: str = "embeddinggemma",
        cache_dir: Optional[Path] = None,
        chunk_size: int = 2000,
        batch_size: int = 10,
        timeout: Optional[float] = None,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        restart_ollama_on_timeout: bool = True
    ):
        """Initialize embedding client.
        
        Args:
            config: LLMConfig instance. If None, loads from environment.
            embedding_model: Name of embedding model to use (default: "embeddinggemma").
            cache_dir: Directory for caching embeddings (default: None, no caching).
            chunk_size: Maximum tokens per chunk for text splitting (default: 2000).
            batch_size: Number of texts to process in each batch (default: 10).
            timeout: Request timeout in seconds (default: uses config.timeout).
            retry_attempts: Number of retry attempts for failed requests (default: 3).
            retry_delay: Initial delay between retries in seconds (default: 2.0, exponential backoff).
            restart_ollama_on_timeout: Whether to attempt Ollama restart on timeout (default: True).
        """
        self.config = config or LLMConfig.from_env()
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.timeout = timeout if timeout is not None else self.config.timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.restart_ollama_on_timeout = restart_ollama_on_timeout and OLLAMA_UTILS_AVAILABLE
        
        # Track consecutive timeouts to detect hung state
        self._consecutive_timeouts = 0
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _log_and_print(
        self,
        message: str,
        level: str = "info",
        flush: bool = True
    ) -> None:
        """Log message and print to stdout for immediate visibility.
        
        Args:
            message: Message to log and print.
            level: Log level ("info", "warning", "error", "debug"). Default: "info".
            flush: Whether to flush stdout after printing. Default: True.
        """
        # Print for immediate visibility
        print(message)
        
        # Log for persistent record
        if level == "info":
            logger.info(message.strip())
        elif level == "warning":
            logger.warning(message.strip())
        elif level == "error":
            logger.error(message.strip())
        elif level == "debug":
            logger.debug(message.strip())
        
        # Flush stdout to ensure immediate output
        if flush:
            sys.stdout.flush()
    
    def check_connection(self, timeout: float = 5.0) -> Tuple[bool, Optional[str]]:
        """Check if Ollama server is accessible.
        
        Args:
            timeout: Connection timeout in seconds.
            
        Returns:
            Tuple of (is_available: bool, error_message: str | None).
            - is_available: True if Ollama is accessible
            - error_message: Error description if unavailable, None if available
        """
        try:
            url = f"{self.config.base_url}/api/tags"
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                logger.debug(f"Ollama connection check successful at {self.config.base_url}")
                return (True, None)
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.warning(f"Ollama connection check failed: {error_msg}")
                return (False, error_msg)
        except Timeout:
            error_msg = f"Timeout after {timeout}s"
            logger.debug(f"Ollama connection check timeout: {error_msg}")
            return (False, error_msg)
        except RequestsConnectionError as e:
            error_msg = f"Connection error: {e}"
            logger.debug(f"Ollama connection check failed: {error_msg}")
            return (False, error_msg)
        except Exception as e:
            error_msg = f"Request error: {e}"
            logger.warning(f"Ollama connection check failed: {error_msg}")
            return (False, error_msg)
    
    def check_embedding_endpoint(self, timeout: float = 10.0) -> Tuple[bool, Optional[str]]:
        """Check if Ollama embedding endpoint is functional.
        
        Tests the actual embedding endpoint with a small text sample to detect
        hung states where the general API responds but embeddings are stuck.
        This is critical for detecting hung Ollama instances.
        
        Args:
            timeout: Request timeout in seconds (default: 10.0).
            
        Returns:
            Tuple of (is_available: bool, error_message: str | None).
            - is_available: True if embedding endpoint is functional
            - error_message: Error description if unavailable, None if available
            
        Example:
            >>> client = EmbeddingClient()
            >>> is_ok, error = client.check_embedding_endpoint()
            >>> if not is_ok:
            ...     print(f"Embedding endpoint hung: {error}")
        """
        if OLLAMA_UTILS_AVAILABLE:
            try:
                from infrastructure.llm.utils.ollama import test_embedding_endpoint
                return test_embedding_endpoint(
                    base_url=self.config.base_url,
                    model=self.embedding_model,
                    timeout=timeout
                )
            except Exception as e:
                error_msg = f"Failed to test embedding endpoint: {e}"
                logger.warning(error_msg)
                return (False, error_msg)
        else:
            # Fallback: test directly
            try:
                url = f"{self.config.base_url}/api/embed"
                payload = {
                    "model": self.embedding_model,
                    "input": "test"  # Small test text
                }
                
                logger.debug(f"Testing embedding endpoint at {url} (timeout: {timeout}s)")
                response = requests.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                
                result = response.json()
                embeddings_list = result.get("embeddings", [])
                if not embeddings_list:
                    embeddings_list = [result.get("embedding", [])]
                
                if embeddings_list and len(embeddings_list[0]) > 0:
                    logger.debug(f"Embedding endpoint check successful (dimension: {len(embeddings_list[0])})")
                    return (True, None)
                else:
                    error_msg = "Empty embedding returned"
                    logger.warning(f"Embedding endpoint check failed: {error_msg}")
                    return (False, error_msg)
            except Timeout:
                error_msg = f"Timeout after {timeout}s - embedding endpoint may be hung"
                logger.warning(f"Embedding endpoint check failed: {error_msg}")
                return (False, error_msg)
            except RequestsConnectionError as e:
                error_msg = f"Connection error: {e}"
                logger.warning(f"Embedding endpoint check failed: {error_msg}")
                return (False, error_msg)
            except Exception as e:
                error_msg = f"Request error: {e}"
                logger.warning(f"Embedding endpoint check failed: {error_msg}")
                return (False, error_msg)
    
    def _ensure_ollama_ready(self) -> bool:
        """Ensure Ollama is running and ready, restarting if needed.
        
        Performs comprehensive validation:
        1. Checks server connection
        2. Verifies embedding model is available
        3. Tests actual embedding generation
        
        Returns:
            True if Ollama is ready and functional, False otherwise.
        """
        if not OLLAMA_UTILS_AVAILABLE:
            logger.warning("Ollama utilities not available - cannot perform health checks")
            return False
        
        # Step 1/3: Quick connection check first
        logger.info("Step 1/3: Checking Ollama connection...")
        is_available, error = self.check_connection(timeout=2.0)
        if not is_available:
            logger.warning(f"Ollama not responding: {error}. Attempting restart...")
            
            # Attempt restart
            success, status_msg = restart_ollama_server(
                base_url=self.config.base_url,
                kill_existing=True,
                wait_seconds=5.0  # Increased wait time for model to be ready
            )
            
            if not success:
                logger.warning(f"Failed to restart Ollama: {status_msg}")
                logger.warning("Continuing anyway - will attempt restart on first embedding if needed")
                return False  # Non-blocking - allow process to continue
            
            logger.info(f"Ollama restarted: {status_msg}")
            # Re-check connection after restart
            is_available, error = self.check_connection(timeout=5.0)
            if not is_available:
                logger.warning(f"Ollama still not responding after restart: {error}")
                logger.warning("Continuing anyway - will attempt restart on first embedding if needed")
                return False  # Non-blocking
        
        logger.info("  ✓ Connection OK")
        
        # Step 2/3: Verify embedding model is available
        logger.info(f"Step 2/3: Verifying embedding model '{self.embedding_model}' is available...")
        try:
            from infrastructure.llm.utils.ollama import get_model_names
            available_models = get_model_names(self.config.base_url)
            
            # Check if model exists (exact match or partial)
            model_found = False
            for model in available_models:
                if model == self.embedding_model or self.embedding_model in model:
                    model_found = True
                    logger.info(f"  ✓ Embedding model found: {model}")
                    break
            
            if not model_found:
                logger.warning(
                    f"Embedding model '{self.embedding_model}' not found. "
                    f"Available models: {', '.join(available_models[:5])}"
                    + (f" and {len(available_models) - 5} more" if len(available_models) > 5 else "")
                )
                logger.warning(f"Install with: ollama pull {self.embedding_model}")
                logger.warning("Continuing anyway - model may load on first use")
                # Non-blocking - continue
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
            logger.warning("Continuing anyway - model check failed but may still work")
            # Continue anyway - might work
        
        # Step 3/3: Test actual embedding generation with small text (short timeout)
        logger.info("Step 3/3: Testing embedding generation functionality...")
        test_timeout = 10.0  # Short timeout for test (vs 60s for real embeddings)
        try:
            test_text = "test"
            # Use direct request with short timeout for test
            url = f"{self.config.base_url}/api/embed"
            payload = {
                "model": self.embedding_model,
                "input": test_text
            }
            
            logger.debug(f"  Sending test embedding request (timeout: {test_timeout}s)...")
            response = requests.post(url, json=payload, timeout=test_timeout)
            response.raise_for_status()
            
            result = response.json()
            embeddings_list = result.get("embeddings", [])
            if not embeddings_list:
                embeddings_list = [result.get("embedding", [])]
            
            if embeddings_list and len(embeddings_list[0]) > 0:
                logger.info(f"  ✓ Embedding test successful (dimension: {len(embeddings_list[0])})")
                return True
            else:
                logger.warning("Embedding test returned empty result")
                logger.warning("Continuing anyway - test failed but may work for real embeddings")
                return False  # Non-blocking
        except Timeout:
            logger.warning(f"Pre-flight test timed out after {test_timeout}s - Ollama may be slow")
            logger.warning("Continuing anyway - will use full timeout for real embeddings")
            return False  # Non-blocking - allow process to continue
        except Exception as e:
            logger.warning(f"Embedding test failed: {e}")
            logger.warning("Continuing anyway - test failed but may work for real embeddings")
            return False  # Non-blocking
    
    def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text with retry logic and health checks.
        
        Performs comprehensive error handling:
        - Quick health check before request (2s timeout)
        - Automatic Ollama restart if health check fails
        - Retry with exponential backoff on failures
        - Hung Ollama detection via embedding endpoint test
        - Force restart if embedding endpoint is hung but general API works
        - RequestMonitor for long operations (timeout > 30s) with adaptive heartbeat intervals
        
        **Progress Visibility:**
        - For requests with timeout > 30s, a RequestMonitor is automatically started
        - Heartbeat interval is adaptive based on text length:
          * Very long documents (>50K chars): 5-second intervals
          * Medium documents (10K-50K chars): 10-second intervals
          * Shorter documents: 20-second intervals
        - Timeout warnings are logged at 50%, 75%, and 90% of timeout elapsed
        - All progress messages are both printed (immediate) and logged (persistent)
        
        Args:
            text: Text to embed.
            use_cache: Whether to use cached embeddings if available.
            
        Returns:
            Embedding vector as numpy array.
            
        Raises:
            LLMConnectionError: If connection to Ollama fails after all retries.
            LLMError: If embedding generation fails after all retries.
            
        Example:
            >>> client = EmbeddingClient()
            >>> embedding = client.generate_embedding("Machine learning is fascinating")
            >>> print(f"Embedding dimension: {len(embedding)}")
            
        Note:
            On timeout, automatically tests embedding endpoint and force restarts
            Ollama if the endpoint is hung (even if general API responds).
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache
        if use_cache and self.cache_dir:
            cached = self._load_from_cache(text)
            if cached is not None:
                logger.debug("Using cached embedding")
                return cached
        
        # Quick health check before embedding request (prevents waiting full timeout if hung)
        is_available, error = self.check_connection(timeout=2.0)
        if not is_available:
            logger.warning(f"Quick health check failed: {error}. Attempting restart before embedding...")
            if self.restart_ollama_on_timeout:
                if self._ensure_ollama_ready():
                    logger.info("Ollama restarted after health check failure, proceeding with embedding")
                else:
                    logger.warning("Ollama restart failed, will attempt embedding anyway")
        
        # Generate embedding with retry logic
        url = f"{self.config.base_url}/api/embed"
        
        payload = {
            "model": self.embedding_model,
            "input": text
        }
        
        last_error = None
        restart_attempted = False
        
        for attempt in range(self.retry_attempts + 1):
            try:
                # On retry after timeout, check/restart Ollama if enabled
                if attempt > 0:
                    # If we just restarted, skip the backoff delay (already waited during restart)
                    if not restart_attempted:
                        wait_time = min(self.retry_delay * (2 ** (attempt - 1)), 30.0)  # Exponential backoff, max 30s
                        logger.info(
                            f"Retrying embedding generation (attempt {attempt + 1}/{self.retry_attempts + 1}) "
                            f"after {wait_time:.1f}s... (text length: {len(text)})"
                        )
                        time.sleep(wait_time)
                    else:
                        # Restart just happened, no need for additional delay
                        logger.info(
                            f"Retrying embedding generation after restart (attempt {attempt + 1}/{self.retry_attempts + 1})... "
                            f"(text length: {len(text)})"
                        )
                    
                    # Quick health check before retry
                    is_available, error = self.check_connection(timeout=2.0)
                    if not is_available:
                        logger.warning(f"Health check before retry failed: {error}")
                    
                    # On timeout errors, attempt Ollama restart if enabled
                    if isinstance(last_error, Timeout) and self.restart_ollama_on_timeout:
                        # Increment consecutive timeout counter
                        self._consecutive_timeouts += 1
                        
                        # Force restart if 2+ consecutive timeouts (even if general API works)
                        if self._consecutive_timeouts >= 2:
                            logger.warning(
                                f"Multiple consecutive timeouts detected ({self._consecutive_timeouts}) - "
                                "forcing Ollama restart regardless of API status"
                            )
                            restart_attempted = True  # Force restart attempt
                        
                        # Also attempt restart on first timeout if not already attempted
                        if not restart_attempted:
                            logger.warning("Timeout detected - Ollama may be hung. Attempting force restart...")
                            # Use very short timeout just to confirm hung state (we already know from the timeout)
                            embedding_ok, embedding_error = self.check_embedding_endpoint(timeout=2.0)
                            if not embedding_ok:
                                logger.warning(f"Embedding endpoint confirmed hung: {embedding_error}. Force restarting...")
                                restart_attempted = True
                            else:
                                # Endpoint responded quickly, but we still timed out on the actual request - restart anyway
                                logger.warning("Embedding endpoint responded to test but timed out on actual request. Restarting anyway...")
                                restart_attempted = True
                        
                        # Force kill and restart Ollama if needed
                        if restart_attempted and OLLAMA_UTILS_AVAILABLE:
                            try:
                                from infrastructure.llm.utils.ollama import restart_ollama_server
                                # Force kill existing process, skip pre-test since we already know it's hung
                                success, status_msg = restart_ollama_server(
                                    base_url=self.config.base_url,
                                    kill_existing=True,
                                    wait_seconds=8.0,  # Longer wait after force kill
                                    test_embedding=True,  # Test embedding endpoint after restart to verify recovery
                                    skip_pre_test=True,  # Skip pre-restart test since we already know endpoint is hung
                                    embedding_model=self.embedding_model
                                )
                                if success:
                                    logger.info(f"Ollama force restarted successfully: {status_msg}")
                                    logger.info(f"Proceeding with embedding retry after restart (attempt {attempt + 1}/{self.retry_attempts + 1})...")
                                    # Reset consecutive timeout counter after successful restart
                                    self._consecutive_timeouts = 0
                                    # Wait a bit more for model to be ready
                                    time.sleep(2.0)
                                    logger.info("Model ready, attempting embedding generation...")
                                else:
                                    logger.warning(f"Ollama force restart failed: {status_msg}")
                            except Exception as e:
                                logger.warning(f"Failed to force restart Ollama: {e}")
                        
                        # Also try the standard health check if restart wasn't attempted
                        if not restart_attempted:
                            if self._ensure_ollama_ready():
                                restart_attempted = True
                                self._consecutive_timeouts = 0  # Reset on successful health check
                                logger.info("Ollama health check successful after timeout")
                                logger.info(f"Proceeding with embedding retry after health check (attempt {attempt + 1}/{self.retry_attempts + 1})...")
                            else:
                                logger.warning("Ollama restart failed or not available, continuing with retry...")
                
                # Log before making the request (especially important after restart)
                if attempt > 0 or restart_attempted:
                    logger.info(
                        f"Sending embedding request to Ollama "
                        f"(text length: {len(text):,} chars, timeout: {self.timeout:.1f}s, "
                        f"attempt: {attempt + 1}/{self.retry_attempts + 1})..."
                    )
                
                # Track request start time for timeout detection after restart
                request_start_time = time.time()
                still_processing_logged = False
                
                # Validate Ollama is ready before sending request
                is_ready, ready_error = self.check_connection(timeout=2.0)
                if not is_ready:
                    logger.warning(f"Ollama connection check before request failed: {ready_error}")
                    print(f"  ⚠ Warning: Ollama connection check failed, but proceeding with request...")
                else:
                    logger.debug("Ollama connection verified before request")
                
                # Immediate confirmation that request is being sent
                self._log_and_print(
                    f"  → Sending embedding request to Ollama (text: {len(text):,} chars, timeout: {self.timeout:.1f}s, model: {self.embedding_model})...",
                    level="info"
                )
                
                # Calculate adaptive heartbeat interval based on text length
                text_len = len(text)
                if text_len > 50000:
                    heartbeat_interval = 5.0  # Very long documents: 5s intervals
                elif text_len > 10000:
                    heartbeat_interval = 10.0  # Medium documents: 10s intervals
                else:
                    heartbeat_interval = 20.0  # Shorter documents: 20s intervals
                
                # Start request monitor for long operations (only if timeout > 30s to avoid overhead)
                monitor = None
                if self.timeout > 30.0:
                    monitor = RequestMonitor(
                        timeout=self.timeout,
                        text_length=len(text),
                        heartbeat_interval=heartbeat_interval
                    )
                    self._log_and_print(
                        f"  → Request monitor started (heartbeat every {heartbeat_interval:.0f}s)",
                        level="info"
                    )
                    monitor.start()
                
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    # Confirm request was successfully sent and response received
                    logger.debug("Embedding request completed successfully")
                finally:
                    # Stop monitoring when request completes (or fails)
                    if monitor:
                        monitor.stop()
                
                # Check if request took a while (especially after restart)
                request_time = time.time() - request_start_time
                if request_time > 10.0 and restart_attempted and not still_processing_logged:
                    logger.info(f"Embedding request completed after restart (took {request_time:.1f}s)")
                
                result = response.json()
                # Ollama returns "embeddings" (plural) as array of arrays
                # For single input, it's [[...]], so take first element
                embeddings_list = result.get("embeddings", [])
                if not embeddings_list:
                    # Fallback to singular "embedding" for compatibility
                    embeddings_list = [result.get("embedding", [])]
                
                if not embeddings_list or len(embeddings_list) == 0:
                    raise LLMError(f"Empty embedding returned from model {self.embedding_model}")
                
                # Take first embedding (for single input, this is the only one)
                embedding = np.array(embeddings_list[0])
                
                if len(embedding) == 0:
                    raise LLMError(f"Empty embedding returned from model {self.embedding_model}")
                
                # Cache result
                if use_cache and self.cache_dir:
                    self._save_to_cache(text, embedding)
                
                # Reset consecutive timeout counter on success
                self._consecutive_timeouts = 0
                
                if attempt > 0:
                    logger.info(f"Embedding generation succeeded on retry {attempt + 1}")
                
                return embedding
                
            except Timeout as e:
                last_error = e
                error_msg = f"Timeout after {self.timeout}s (text length: {len(text)})"
                logger.warning(f"Embedding generation timeout (attempt {attempt + 1}/{self.retry_attempts + 1}): {error_msg}")
                if attempt == self.retry_attempts:
                    # Final attempt failed
                    raise LLMConnectionError(
                        f"Failed to generate embedding after {self.retry_attempts + 1} attempts: {error_msg}",
                        context={
                            "url": url,
                            "model": self.embedding_model,
                            "text_length": len(text),
                            "timeout": self.timeout
                        }
                    ) from e
                    
            except RequestsConnectionError as e:
                last_error = e
                error_msg = f"Connection error: {e}"
                logger.warning(f"Embedding generation connection error (attempt {attempt + 1}/{self.retry_attempts + 1}): {error_msg}")
                if attempt == self.retry_attempts:
                    # Final attempt failed
                    raise LLMConnectionError(
                        f"Failed to connect to Ollama after {self.retry_attempts + 1} attempts: {error_msg}",
                        context={"url": url, "model": self.embedding_model, "text_length": len(text)}
                    ) from e
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                error_msg = f"Request error: {e}"
                logger.warning(f"Embedding generation request error (attempt {attempt + 1}/{self.retry_attempts + 1}): {error_msg}")
                if attempt == self.retry_attempts:
                    # Final attempt failed
                    raise LLMConnectionError(
                        f"Failed to generate embedding after {self.retry_attempts + 1} attempts: {error_msg}",
                        context={"url": url, "model": self.embedding_model, "text_length": len(text)}
                    ) from e
                    
            except Exception as e:
                # Non-retryable errors (e.g., invalid response format)
                raise LLMError(
                    f"Failed to generate embedding: {e}",
                    context={"model": self.embedding_model, "text_length": len(text)}
                ) from e
        
        # Should not reach here, but handle just in case
        raise LLMError(
            f"Failed to generate embedding after {self.retry_attempts + 1} attempts",
            context={"model": self.embedding_model, "text_length": len(text)}
        )
    
    def test_embedding_generation(
        self,
        test_text: str = "This is a test embedding"
    ) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """Generate a test embedding to verify Ollama functionality.
        
        Useful for verifying that Ollama is working correctly after a restart.
        Generates embedding for a simple test string and logs the result.
        
        Args:
            test_text: Text to use for test embedding (default: "This is a test embedding").
            
        Returns:
            Tuple of (success: bool, embedding: np.ndarray | None, message: str | None).
            - success: True if embedding was generated successfully
            - embedding: Embedding vector if successful, None otherwise
            - message: Status message describing the result
            
        Example:
            >>> client = EmbeddingClient()
            >>> success, embedding, msg = client.test_embedding_generation()
            >>> if success:
            ...     print(f"Test passed: {msg}")
            ...     print(f"Embedding dimension: {len(embedding)}")
        """
        logger.info("=" * 60)
        logger.info("TEST EMBEDDING GENERATION")
        logger.info("=" * 60)
        logger.info(f"Test text: '{test_text}'")
        logger.info(f"Model: {self.embedding_model}")
        logger.info(f"Timeout: {self.timeout:.1f}s")
        
        start_time = time.time()
        
        try:
            embedding = self.generate_embedding(test_text, use_cache=False)
            elapsed_time = time.time() - start_time
            
            # Log embedding details
            embedding_dim = len(embedding)
            embedding_norm = float(np.linalg.norm(embedding))
            
            # Get first 10 values for logging
            first_values = embedding[:10].tolist()
            
            logger.info("")
            logger.info("✓ Test embedding generated successfully")
            logger.info(f"  Dimension: {embedding_dim}")
            logger.info(f"  Norm: {embedding_norm:.6f}")
            logger.info(f"  First 10 values: {[f'{v:.6f}' for v in first_values]}")
            logger.info(f"  Generation time: {elapsed_time:.2f}s")
            logger.info("=" * 60)
            logger.info("")
            
            message = (
                f"Test embedding successful: dimension={embedding_dim}, "
                f"norm={embedding_norm:.6f}, time={elapsed_time:.2f}s"
            )
            
            # Reset consecutive timeout counter on successful test
            self._consecutive_timeouts = 0
            
            return (True, embedding, message)
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_type = type(e).__name__
            error_msg = str(e)
            
            logger.error("")
            logger.error("✗ Test embedding generation failed")
            logger.error(f"  Error type: {error_type}")
            logger.error(f"  Error message: {error_msg}")
            logger.error(f"  Time elapsed: {elapsed_time:.2f}s")
            logger.error("=" * 60)
            logger.error("")
            
            message = f"Test embedding failed: {error_type} - {error_msg}"
            
            return (False, None, message)
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed.
            use_cache: Whether to use cached embeddings if available.
            show_progress: Whether to log progress.
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        total = len(texts)
        start_time = time.time()
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            
            if show_progress:
                logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            for j, text in enumerate(batch):
                chunk_num = i + j + 1
                text_len = len(text)
                # Estimate time per chunk based on text length (rough: 0.025s per char)
                estimated_time = max(1.0, text_len / 40.0)
                
                if show_progress and total > 1:
                    chunk_msg = (
                        f"  → Processing chunk {chunk_num}/{total} "
                        f"({text_len:,} chars, ~{estimated_time:.1f}s estimated)"
                    )
                    # Note: generate_embeddings_batch doesn't have self context for _log_and_print
                    # So we use print+logger directly here
                    print(chunk_msg)
                    logger.info(chunk_msg.strip())
                    sys.stdout.flush()
                
                try:
                    embedding = self.generate_embedding(text, use_cache=use_cache)
                    embeddings.append(embedding)
                    
                    if show_progress and total > 1:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / chunk_num if chunk_num > 0 else 0
                        remaining = total - chunk_num
                        eta = avg_time * remaining if avg_time > 0 else 0
                        completion_msg = (
                            f"  ✓ Chunk {chunk_num}/{total} completed "
                            f"(avg: {avg_time:.1f}s/chunk, ETA: {eta:.0f}s remaining)"
                        )
                        # Note: generate_embeddings_batch doesn't have self context for _log_and_print
                        # So we use print+logger directly here
                        print(completion_msg)
                        logger.info(completion_msg.strip())
                        sys.stdout.flush()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for chunk {chunk_num}/{total} (length={text_len:,}): {e}")
                    # Use zero vector as fallback
                    if embeddings:
                        fallback = np.zeros_like(embeddings[0])
                    else:
                        # First embedding failed, need to get dimension from API
                        try:
                            sample = self.generate_embedding("sample", use_cache=False)
                            fallback = np.zeros_like(sample)
                        except:
                            fallback = np.zeros(768)  # Default embeddinggemma dimension
                    embeddings.append(fallback)
            
            # Small delay between batches to avoid overwhelming the API
            if i + self.batch_size < total:
                time.sleep(0.1)
        
        return np.array(embeddings)
    
    def chunk_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        overlap: int = 200
    ) -> List[str]:
        """Split text into chunks for embedding.
        
        Uses simple character-based chunking with overlap. For more sophisticated
        token-based chunking, consider using a tokenizer library.
        
        Args:
            text: Text to chunk.
            max_tokens: Maximum tokens per chunk (default: self.chunk_size).
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            List of text chunks.
        """
        if max_tokens is None:
            max_tokens = self.chunk_size
        
        # Simple character-based chunking
        # Approximate: 1 token ≈ 4 characters (conservative estimate)
        chunk_size_chars = max_tokens * 4
        
        if len(text) <= chunk_size_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size_chars
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within last 20% of chunk
                search_start = max(start, end - chunk_size_chars // 5)
                for i in range(end - 1, search_start, -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def embed_document(
        self,
        text: str,
        aggregation: str = "mean",
        use_cache: bool = True
    ) -> np.ndarray:
        """Embed a full document by chunking and aggregating.
        
        Automatically chunks large texts (>4000 chars) to avoid timeouts.
        Uses adaptive timeouts based on text/chunk length:
        - Very short texts (<100 chars): 30s minimum (hung detection)
        - Longer texts: min(config_timeout, max(30s, text_length/40))
        
        **Chunk Processing:**
        - Documents are automatically chunked if they exceed chunk_size tokens
        - For very long documents (>100K chars), chunk size is reduced by half
        - Each chunk is processed sequentially with progress logging
        - Chunk embeddings are aggregated using the specified method (default: mean)
        - Progress is logged for each chunk: start, completion, and ETA
        
        **Progress Visibility:**
        - Chunk count and average chunk size are logged immediately
        - Per-chunk timeout is calculated and displayed
        - Each chunk shows: chunk number, size, estimated time, and completion status
        - Overall progress includes average time per chunk and ETA
        
        Args:
            text: Full document text.
            aggregation: Aggregation method ("mean", "max", "sum") (default: "mean").
            use_cache: Whether to use cached embeddings if available.
            
        Returns:
            Document-level embedding vector.
            
        Example:
            >>> client = EmbeddingClient()
            >>> # Large document automatically chunked
            >>> embedding = client.embed_document(long_text, aggregation="mean")
            >>> print(f"Document embedding dimension: {len(embedding)}")
            
        Note:
            Chunking uses sentence boundaries when possible to preserve context.
            Overlap between chunks ensures continuity.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache for full document
        if use_cache and self.cache_dir:
            cached = self._load_from_cache(text)
            if cached is not None:
                logger.info("Using cached document embedding")
                return cached
        
        # Chunk text if needed (use smaller chunks to avoid timeouts)
        # For very long documents (>100K chars), use more aggressive chunking (half chunk size)
        # This reduces per-chunk processing time and improves checkpoint granularity
        text_length = len(text)
        effective_chunk_size = self.chunk_size
        # Check if we need to reduce chunk size for very long documents
        # This threshold can be configured via embedding_chunk_size_reduction_threshold
        # Default is 100000 chars - reduce chunk size by half for documents exceeding this
        chunk_size_reduction_threshold = getattr(self, '_chunk_size_reduction_threshold', 100000)
        if text_length > chunk_size_reduction_threshold:
            effective_chunk_size = max(500, self.chunk_size // 2)  # Half chunk size, minimum 500 tokens
            logger.debug(f"Using reduced chunk size {effective_chunk_size} for long document ({text_length:,} chars)")
        
        chunks = self.chunk_text(text, max_tokens=min(effective_chunk_size, 1000))
        
        if len(chunks) == 1:
            # Single chunk, no aggregation needed
            # Use adaptive timeout for single chunk based on text length
            # Unified formula: min(config_timeout, max(30s, text_length/40))
            # For very short texts (<100 chars), use minimum 30s (hung detection)
            text_len = len(chunks[0])
            if text_len < 100:
                # Very short text - if timeout, likely hung (not slow)
                chunk_timeout = min(self.timeout, 30.0)
            else:
                # Scale with text length: 0.025s per char (1/40), minimum 30s, maximum configured timeout
                chunk_timeout = min(self.timeout, max(30.0, text_len / 40.0))
            original_timeout = self.timeout
            self.timeout = chunk_timeout
            try:
                embedding = self.generate_embedding(chunks[0], use_cache=use_cache)
            finally:
                self.timeout = original_timeout
        else:
            # Generate embeddings for all chunks
            self._log_and_print(
                f"  → Document will be processed in {len(chunks)} chunks (text length: {len(text):,} chars)",
                level="info"
            )
            logger.info(f"Embedding document with {len(chunks)} chunks (text length: {len(text):,} chars)")
            avg_chunk_size = len(text) / len(chunks)
            self._log_and_print(
                f"  → Average chunk size: {avg_chunk_size:,.0f} chars",
                level="info"
            )
            logger.info(f"  Average chunk size: {avg_chunk_size:,.0f} chars")
            # Use adaptive timeout per chunk
            # Unified formula: min(config_timeout, max(30s, avg_chunk_size/40))
            original_timeout = self.timeout
            if avg_chunk_size < 100:
                # Very short chunks - if timeout, likely hung (not slow)
                chunk_timeout = min(self.timeout, 30.0)
            else:
                # Scale with chunk size: 0.025s per char (1/40), minimum 30s, maximum configured timeout
                chunk_timeout = min(self.timeout, max(30.0, avg_chunk_size / 40.0))
            self.timeout = chunk_timeout
            self._log_and_print(
                f"  → Per-chunk timeout: {chunk_timeout:.1f}s",
                level="info"
            )
            logger.info(f"  Per-chunk timeout: {chunk_timeout:.1f}s")
            
            try:
                chunk_embeddings = self.generate_embeddings_batch(
                    chunks,
                    use_cache=use_cache,
                    show_progress=True
                )
            finally:
                self.timeout = original_timeout
            
            # Aggregate chunk embeddings
            if aggregation == "mean":
                embedding = np.mean(chunk_embeddings, axis=0)
            elif aggregation == "max":
                embedding = np.max(chunk_embeddings, axis=0)
            elif aggregation == "sum":
                embedding = np.sum(chunk_embeddings, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Cache result
        if use_cache and self.cache_dir:
            self._save_to_cache(text, embedding)
        
        return embedding
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text.
        
        Args:
            text: Text to generate key for.
            
        Returns:
            Cache key string.
        """
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _cache_path(self, text: str) -> Path:
        """Get cache file path for text.
        
        Args:
            text: Text to get cache path for.
            
        Returns:
            Path to cache file.
        """
        if not self.cache_dir:
            raise ValueError("Cache directory not configured")
        
        key = self._cache_key(text)
        return self.cache_dir / f"{key}.json"
    
    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache.
        
        Args:
            text: Text to load embedding for.
            
        Returns:
            Cached embedding or None if not found.
        """
        if not self.cache_dir:
            return None
        
        cache_path = self._cache_path(text)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                embedding = np.array(data.get("embedding", []))
                return embedding if len(embedding) > 0 else None
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache.
        
        Args:
            text: Original text.
            embedding: Embedding vector to cache.
        """
        if not self.cache_dir:
            return
        
        cache_path = self._cache_path(text)
        
        try:
            data = {
                "embedding": embedding.tolist(),
                "model": self.embedding_model,
                "dimension": len(embedding)
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

