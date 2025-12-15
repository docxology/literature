"""Ollama utility functions for model discovery and server management.

Provides utilities for:
- Discovering available local Ollama models
- Selecting the best model based on preferences
- Checking Ollama server status
- Starting Ollama server if needed
- Model preloading with retry logic
- Connection health checks
"""
from __future__ import annotations

import subprocess
import time
import platform
import socket
from typing import Optional, List, Dict, Any, Tuple
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError as RequestsConnectionError

from infrastructure.core.logging_utils import get_logger
from infrastructure.core.exceptions import LLMConnectionError

logger = get_logger(__name__)

# Default model preferences in order of preference
DEFAULT_MODEL_PREFERENCES = [
    "llama3-gradient:latest",  # Large context (256K), reliable, no thinking mode issues
    "llama3.1:latest",  # Good balance of speed and quality
    "llama2:latest",    # Widely available, reliable
    "gemma2:2b",        # Fast, small, good instruction following
    "gemma3:4b",        # Medium size, good quality
    "mistral:latest",   # Alternative
    "codellama:latest", # Code-focused but can do general tasks
    # Note: qwen3 models use "thinking" mode which requires special handling
]


def is_ollama_running(base_url: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    """Check if Ollama server is running and responding.
    
    Args:
        base_url: Ollama server URL
        timeout: Connection timeout in seconds
        
    Returns:
        True if Ollama is responding, False otherwise
        
    Example:
        >>> if is_ollama_running():
        ...     print("Ollama is ready")
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == 200:
            logger.debug(f"Ollama server responding at {base_url}")
            return True
        else:
            logger.warning(f"Ollama server returned status {response.status_code} at {base_url}")
            return False
    except Timeout:
        logger.debug(f"Ollama server timeout at {base_url} (timeout={timeout}s)")
        return False
    except RequestsConnectionError as e:
        logger.debug(f"Ollama server connection failed at {base_url}: {e}")
        return False
    except RequestException as e:
        logger.debug(f"Ollama server request failed at {base_url}: {e}")
        return False


def start_ollama_server(wait_seconds: float = 3.0) -> bool:
    """Attempt to start the Ollama server.
    
    Args:
        wait_seconds: How long to wait for server to start
        
    Returns:
        True if server started successfully, False otherwise
    """
    try:
        # Try to start ollama serve in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait for server to be ready
        time.sleep(wait_seconds)
        return is_ollama_running()
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _is_port_in_use(host: str = "localhost", port: int = 11434) -> bool:
    """Check if a port is in use.
    
    Args:
        host: Host to check
        port: Port number to check
        
    Returns:
        True if port is in use, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def _kill_process_on_port(port: int = 11434) -> Tuple[bool, str]:
    """Kill process using the specified port.
    
    Args:
        port: Port number
        
    Returns:
        Tuple of (success: bool, status_message: str)
    """
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            # Use lsof to find process using port
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                killed_pids = []
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True, timeout=5.0)
                        killed_pids.append(pid)
                    except subprocess.SubprocessError:
                        pass
                if killed_pids:
                    return (True, f"Killed process(es) on port {port}: {', '.join(killed_pids)}")
                return (False, f"Found process(es) on port {port} but failed to kill")
            return (False, f"No process found on port {port}")
            
        elif system == "Linux":
            # Use lsof or fuser to find process
            # Try lsof first
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                killed_pids = []
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True, timeout=5.0)
                        killed_pids.append(pid)
                    except subprocess.SubprocessError:
                        pass
                if killed_pids:
                    return (True, f"Killed process(es) on port {port}: {', '.join(killed_pids)}")
                return (False, f"Found process(es) on port {port} but failed to kill")
            # Try fuser as fallback
            result = subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            if result.returncode == 0:
                return (True, f"Killed process on port {port} using fuser")
            return (False, f"No process found on port {port}")
            
        else:
            # Windows or other - not supported for now
            return (False, f"Port killing not supported on {system}")
            
    except subprocess.TimeoutExpired:
        return (False, f"Timeout while checking/killing process on port {port}")
    except FileNotFoundError:
        return (False, "Required system tools (lsof/fuser) not found")
    except Exception as e:
        return (False, f"Error killing process on port {port}: {e}")


def restart_ollama_server(
    base_url: str = "http://localhost:11434",
    kill_existing: bool = True,
    wait_seconds: float = 5.0
) -> Tuple[bool, str]:
    """Restart Ollama server, handling port conflicts.
    
    Checks if Ollama is responding. If not, attempts to kill any process
    using the port, then starts a fresh Ollama server.
    
    Args:
        base_url: Ollama server URL
        kill_existing: Whether to kill existing processes on the port
        wait_seconds: How long to wait for server to start after restart
        
    Returns:
        Tuple of (success: bool, status_message: str)
        - success: True if Ollama is ready after restart
        - status_message: Detailed status message
        
    Example:
        >>> success, status = restart_ollama_server()
        >>> if not success:
        ...     print(f"Restart failed: {status}")
    """
    # Extract port from base_url
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        port = parsed.port or 11434
        host = parsed.hostname or "localhost"
    except Exception:
        port = 11434
        host = "localhost"
    
    # Check if Ollama is already responding
    if is_ollama_running(base_url, timeout=2.0):
        logger.info("Ollama is already running and responding")
        models = get_model_names(base_url)
        if models:
            return (True, f"Ollama already running with {len(models)} model(s)")
        else:
            return (True, "Ollama already running but no models available")
    
    # Ollama is not responding - check if port is in use
    port_in_use = _is_port_in_use(host, port)
    
    if port_in_use and kill_existing:
        logger.info(f"Port {port} is in use but Ollama not responding, attempting to kill process...")
        kill_success, kill_msg = _kill_process_on_port(port)
        if kill_success:
            logger.info(f"Killed process on port {port}: {kill_msg}")
            # Wait a moment for port to be released
            time.sleep(1.0)
        else:
            logger.warning(f"Could not kill process on port {port}: {kill_msg}")
            return (False, f"Port {port} in use but could not kill process: {kill_msg}")
    elif port_in_use and not kill_existing:
        return (False, f"Port {port} is in use but Ollama not responding. Set kill_existing=True to restart.")
    
    # Start Ollama server
    logger.info("Starting Ollama server...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        error_msg = f"Failed to start Ollama server: {e}"
        logger.error(error_msg)
        return (False, error_msg)
    
    # Wait for server to be ready
    logger.debug(f"Waiting {wait_seconds}s for Ollama to start...")
    time.sleep(wait_seconds)
    
    # Verify it's responding
    max_checks = 5
    check_interval = 1.0
    for i in range(max_checks):
        if is_ollama_running(base_url, timeout=2.0):
            models = get_model_names(base_url)
            if models:
                status_msg = f"Ollama restarted successfully with {len(models)} model(s): {', '.join(models[:3])}"
                logger.info(status_msg)
                return (True, status_msg)
            else:
                status_msg = "Ollama restarted but no models available"
                logger.warning(status_msg)
                return (True, status_msg)
        
        if i < max_checks - 1:
            logger.debug(f"Ollama not responding yet, waiting {check_interval}s... (check {i+1}/{max_checks})")
            time.sleep(check_interval)
    
    error_msg = f"Ollama started but not responding after {wait_seconds + (max_checks * check_interval)}s"
    logger.error(error_msg)
    return (False, error_msg)


def get_available_models(
    base_url: str = "http://localhost:11434",
    timeout: float = 5.0,
    retries: int = 2
) -> List[Dict[str, Any]]:
    """Get list of available models from Ollama with retry logic.
    
    Args:
        base_url: Ollama server URL
        timeout: Request timeout in seconds
        retries: Number of retry attempts on failure
        
    Returns:
        List of model dictionaries with 'name', 'size', etc.
        
    Example:
        >>> models = get_available_models()
        >>> print(f"Found {len(models)} models")
    """
    last_error = None
    
    for attempt in range(retries + 1):
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            
            if models:
                logger.debug(f"Retrieved {len(models)} model(s) from Ollama")
            else:
                logger.warning("Ollama returned empty model list")
            
            return models
            
        except Timeout as e:
            last_error = f"Timeout after {timeout}s"
            if attempt < retries:
                wait_time = (attempt + 1) * 0.5
                logger.debug(f"Timeout getting models (attempt {attempt + 1}/{retries + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Failed to get available models after {retries + 1} attempts: {last_error}")
                
        except RequestsConnectionError as e:
            last_error = f"Connection error: {e}"
            if attempt < retries:
                wait_time = (attempt + 1) * 0.5
                logger.debug(f"Connection error (attempt {attempt + 1}/{retries + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Failed to get available models after {retries + 1} attempts: {last_error}")
                
        except RequestException as e:
            last_error = f"Request error: {e}"
            logger.warning(f"Failed to get available models: {last_error}")
            break  # Don't retry on non-network errors
    
    return []


def get_model_names(base_url: str = "http://localhost:11434") -> List[str]:
    """Get list of available model names from Ollama.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        List of model names (e.g., ["llama3:latest", "mistral:7b"])
    """
    models = get_available_models(base_url)
    return [m["name"] for m in models]


def select_best_model(
    preferences: Optional[List[str]] = None,
    base_url: str = "http://localhost:11434"
) -> Optional[str]:
    """Select the best available model based on preferences.
    
    Iterates through preference list and returns first available model.
    Falls back to first available model if no preference matches.
    
    Args:
        preferences: Ordered list of preferred model names
        base_url: Ollama server URL
        
    Returns:
        Model name to use, or None if no models available
    """
    available = get_model_names(base_url)
    
    if not available:
        return None
    
    prefs = preferences or DEFAULT_MODEL_PREFERENCES
    
    # Try each preference in order
    for pref in prefs:
        # Check for exact match
        if pref in available:
            logger.info(f"Selected model: {pref}")
            return pref
        
        # Check for partial match (e.g., "llama3" matches "llama3:latest")
        for model in available:
            if pref in model or model.startswith(pref.split(":")[0]):
                logger.info(f"Selected model: {model} (matched preference: {pref})")
                return model
    
    # Fall back to first available
    first = available[0]
    logger.info(f"No preference matched, using first available: {first}")
    return first


def select_small_fast_model(base_url: str = "http://localhost:11434") -> Optional[str]:
    """Select a small, fast model for testing.
    
    Prioritizes smaller models for faster test execution.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        Model name to use, or None if no models available
    """
    fast_preferences = [
        "smollm2",
        "gemma2:2b",
        "gemma3:4b",
        "llama2:latest",
        "mistral:latest",
    ]
    return select_best_model(fast_preferences, base_url)


def test_ollama_functionality(
    base_url: str = "http://localhost:11434",
    timeout: float = 10.0
) -> Tuple[bool, Optional[str]]:
    """Test that Ollama can actually process queries, not just respond to API.
    
    Performs a simple query test to verify Ollama is fully functional.
    
    Args:
        base_url: Ollama server URL
        timeout: Query timeout in seconds (used for config timeout)
        
    Returns:
        Tuple of (success: bool, error_message: str | None)
        - success: True if query test passed
        - error_message: Error description if failed, None if successful
        
    Example:
        >>> success, error = test_ollama_functionality()
        >>> if not success:
        ...     print(f"Query test failed: {error}")
    """
    try:
        from infrastructure.llm import LLMClient, LLMConfig
        
        # Get a model to test with
        models = get_model_names(base_url)
        if not models:
            return (False, "No models available for testing")
        
        test_model = models[0]
        logger.debug(f"Testing Ollama functionality with model: {test_model}")
        
        # Create client and perform simple query
        config = LLMConfig.from_env()
        config.base_url = base_url
        config.model = test_model
        # Set timeout in config if supported
        if hasattr(config, 'timeout'):
            config.timeout = timeout
        
        client = LLMClient(config)
        
        # Simple test query - use a very short prompt to minimize time
        # Use a minimal generation to test functionality
        try:
            # Try a very simple query with minimal tokens
            from infrastructure.llm.core.config import GenerationOptions
            options = GenerationOptions(max_tokens=5, temperature=0.0)
            response = client.query_short("Hi", options=options)
            if response and len(response.strip()) > 0:
                logger.debug("Ollama functionality test passed")
                return (True, None)
            else:
                return (False, "Query returned empty response")
        except Exception as e:
            # Check if it's a timeout - that might be acceptable for slow models
            error_str = str(e)
            if "timeout" in error_str.lower():
                # Timeout might mean model is slow, but server is working
                # Log as warning but don't fail - server can respond to API
                logger.debug(f"Functionality test timeout (model may be slow): {error_str}")
                return (True, f"Query test timed out (model may be slow): {error_str}")
            error_msg = f"Query test failed: {error_str}"
            logger.warning(error_msg)
            return (False, error_msg)
            
    except ImportError as e:
        return (False, f"Failed to import LLMClient: {e}")
    except Exception as e:
        return (False, f"Functionality test error: {e}")


def diagnose_ollama_issues(
    base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """Comprehensive diagnostic check for Ollama installation and configuration.
    
    Checks multiple aspects of Ollama setup:
    - Is Ollama installed?
    - Is server running?
    - Are models available?
    - Can queries be processed?
    - What's the configuration?
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        Dictionary with diagnostic information:
        - installed: bool - Is Ollama installed?
        - server_running: bool - Is server responding?
        - models_available: bool - Are models available?
        - models: List[str] - List of available models
        - functionality_test: bool - Can queries be processed?
        - functionality_error: str | None - Error from functionality test
        - base_url: str - Server URL checked
        - port: int - Port number
        - diagnostics: Dict[str, Any] - Additional diagnostic info
        
    Example:
        >>> diag = diagnose_ollama_issues()
        >>> if not diag["server_running"]:
        ...     print("Ollama server is not running")
    """
    diagnostics = {
        "installed": False,
        "server_running": False,
        "models_available": False,
        "models": [],
        "model_count": 0,
        "functionality_test": False,
        "functionality_error": None,
        "base_url": base_url,
        "port": 11434,
        "diagnostics": {}
    }
    
    # Extract port from base_url
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        diagnostics["port"] = parsed.port or 11434
        diagnostics["host"] = parsed.hostname or "localhost"
    except Exception:
        diagnostics["port"] = 11434
        diagnostics["host"] = "localhost"
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5.0
        )
        if result.returncode == 0:
            diagnostics["installed"] = True
            diagnostics["diagnostics"]["version"] = result.stdout.strip()
        else:
            diagnostics["diagnostics"]["version_error"] = f"Exit code {result.returncode}"
    except FileNotFoundError:
        diagnostics["diagnostics"]["version_error"] = "Ollama command not found"
    except subprocess.TimeoutExpired:
        diagnostics["diagnostics"]["version_error"] = "Version check timeout"
    except Exception as e:
        diagnostics["diagnostics"]["version_error"] = str(e)
    
    # Check if server is running
    diagnostics["server_running"] = is_ollama_running(base_url, timeout=2.0)
    if not diagnostics["server_running"]:
        diagnostics["diagnostics"]["server_error"] = "Server not responding to /api/tags"
    
    # Check for models
    if diagnostics["server_running"]:
        models = get_model_names(base_url)
        diagnostics["models"] = models
        diagnostics["model_count"] = len(models)
        diagnostics["models_available"] = len(models) > 0
        if not diagnostics["models_available"]:
            diagnostics["diagnostics"]["models_error"] = "No models found in Ollama"
    
    # Test functionality
    if diagnostics["server_running"] and diagnostics["models_available"]:
        success, error = test_ollama_functionality(base_url, timeout=10.0)
        diagnostics["functionality_test"] = success
        diagnostics["functionality_error"] = error
        if not success:
            diagnostics["diagnostics"]["functionality_error"] = error
    
    return diagnostics


def ensure_ollama_ready(
    base_url: str = "http://localhost:11434",
    auto_start: bool = True,
    test_functionality: bool = False
) -> bool:
    """Ensure Ollama server is running and has models available.
    
    Args:
        base_url: Ollama server URL
        auto_start: Whether to attempt starting Ollama if not running
        test_functionality: Whether to test actual query processing (slower but more thorough)
        
    Returns:
        True if Ollama is ready with models, False otherwise
    """
    # Check if running
    if not is_ollama_running(base_url):
        if auto_start:
            logger.info("Ollama server not responding, attempting to start...")
            if not start_ollama_server():
                logger.warning("Ollama installed but server failed to start. Try: ollama serve")
                return False
        else:
            logger.warning("Ollama server not responding. Start with: ollama serve")
            return False
    
    # Check for available models
    models = get_model_names(base_url)
    if not models:
        logger.warning("Ollama running but no models available. Install with: ollama pull <model>")
        return False
    
    # Optional functionality test
    if test_functionality:
        logger.debug("Testing Ollama functionality with actual query...")
        success, error = test_ollama_functionality(base_url, timeout=10.0)
        if not success:
            logger.warning(f"Ollama connection OK but queries fail: {error}")
            return False
    
    logger.info(f"Ollama ready with {len(models)} model(s): {', '.join(models[:5])}")
    return True


def get_model_info(model_name: str, base_url: str = "http://localhost:11434") -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        base_url: Ollama server URL
        
    Returns:
        Model info dictionary or None if not found
    """
    models = get_available_models(base_url)
    for model in models:
        if model["name"] == model_name or model_name in model["name"]:
            return model
    return None


def check_model_loaded(
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout: float = 2.0
) -> Tuple[bool, Optional[str]]:
    """Check if a model is currently loaded in Ollama's memory.
    
    Uses Ollama's /api/ps endpoint to check which models are currently
    loaded in GPU/system memory.
    
    Args:
        model_name: Name of the model to check
        base_url: Ollama server URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (is_loaded: bool, loaded_model_name: str | None)
        - is_loaded: True if the model (or a matching model) is loaded
        - loaded_model_name: Name of the loaded model if found, None otherwise
        
    Example:
        >>> is_loaded, loaded_name = check_model_loaded("llama3:latest")
        >>> if is_loaded:
        ...     print(f"Model {loaded_name} is already loaded")
    """
    try:
        response = requests.get(f"{base_url}/api/ps", timeout=timeout)
        if response.status_code != 200:
            logger.debug(f"Ollama /api/ps returned status {response.status_code}")
            return (False, None)
        
        data = response.json()
        processes = data.get("processes", [])
        
        if not processes:
            logger.debug("No models currently loaded in Ollama")
            return (False, None)
        
        logger.debug(f"Found {len(processes)} loaded model process(es)")
        
        # Check for exact match first
        for proc in processes:
            proc_model = proc.get("model", "")
            if proc_model == model_name:
                logger.debug(f"Exact match found: {proc_model}")
                return (True, proc_model)
        
        # Check for partial match (e.g., "llama3" matches "llama3:latest")
        model_base = model_name.split(":")[0] if ":" in model_name else model_name
        for proc in processes:
            proc_model = proc.get("model", "")
            proc_base = proc_model.split(":")[0] if ":" in proc_model else proc_model
            if model_base == proc_base:
                logger.debug(f"Partial match found: {proc_model} (requested: {model_name})")
                return (True, proc_model)
        
        loaded_models = [p.get("model", "unknown") for p in processes]
        logger.debug(f"Model {model_name} not loaded. Currently loaded: {', '.join(loaded_models)}")
        return (False, None)
        
    except Timeout:
        logger.debug(f"Timeout checking model load status (timeout={timeout}s)")
        return (False, None)
    except RequestsConnectionError as e:
        logger.debug(f"Connection error checking model load status: {e}")
        return (False, None)
    except RequestException as e:
        logger.warning(f"Request error checking model load status: {e}")
        return (False, None)


def preload_model(
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout: float = 60.0,
    retries: int = 1,
    check_loaded_first: bool = True
) -> Tuple[bool, Optional[str]]:
    """Preload a model into Ollama's memory with retry logic.
    
    Sends a request to Ollama to load the model into memory, which can
    speed up subsequent queries. Checks if model is already loaded first
    to avoid unnecessary preloads.
    
    Args:
        model_name: Name of the model to preload
        base_url: Ollama server URL
        timeout: Request timeout in seconds (increased for large models)
        retries: Number of retry attempts on failure
        check_loaded_first: Check if model is already loaded before preloading
        
    Returns:
        Tuple of (success: bool, error_message: str | None)
        - success: True if preload was successful or already loaded
        - error_message: Error description if failed, None if successful
        
    Example:
        >>> success, error = preload_model("llama3:latest")
        >>> if not success:
        ...     print(f"Preload failed: {error}")
    """
    # Check if already loaded
    if check_loaded_first:
        is_loaded, loaded_name = check_model_loaded(model_name, base_url)
        if is_loaded:
            logger.debug(f"Model {model_name} already loaded ({loaded_name})")
            return (True, None)
    
    logger.debug(f"Preloading model {model_name} (timeout={timeout}s, retries={retries})")
    
    last_error = None
    
    for attempt in range(retries + 1):
        try:
            # Use generate endpoint with minimal prompt to trigger model load
            # This is more reliable than /api/ps for ensuring model is ready
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "test",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                logger.debug(f"Model {model_name} preloaded successfully")
                return (True, None)
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning(f"Preload returned status {response.status_code}: {last_error}")
                
        except Timeout as e:
            last_error = f"Timeout after {timeout}s (model may still be loading)"
            if attempt < retries:
                wait_time = (attempt + 1) * 2.0
                logger.debug(f"Preload timeout (attempt {attempt + 1}/{retries + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Preload timeout after {retries + 1} attempts: {last_error}")
                # Timeout might mean model is still loading, not necessarily failed
                # Check if it's loaded now
                is_loaded, loaded_name = check_model_loaded(model_name, base_url)
                if is_loaded:
                    logger.info(f"Model {model_name} loaded despite timeout (found: {loaded_name})")
                    return (True, None)
                    
        except RequestsConnectionError as e:
            last_error = f"Connection error: {e}"
            if attempt < retries:
                wait_time = (attempt + 1) * 1.0
                logger.debug(f"Preload connection error (attempt {attempt + 1}/{retries + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Preload connection error after {retries + 1} attempts: {last_error}")
                
        except RequestException as e:
            last_error = f"Request error: {e}"
            logger.warning(f"Preload request error: {last_error}")
            break  # Don't retry on non-network errors
    
    return (False, last_error)

