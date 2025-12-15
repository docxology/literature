"""Load test configuration from YAML file."""
from pathlib import Path
import yaml
from infrastructure.llm.core.config import LLMConfig


def load_test_ollama_config() -> dict:
    """Load Ollama test configuration from YAML."""
    config_path = Path(__file__).parent / "test_ollama_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_test_llm_config() -> LLMConfig:
    """Get LLMConfig configured for tests."""
    config_data = load_test_ollama_config()
    ollama = config_data.get("ollama", {})
    
    return LLMConfig(
        base_url=ollama.get("host", "http://localhost:11434"),
        timeout=ollama.get("timeout", 120.0),
        default_model=ollama.get("default_model", "gemma3:4b"),
        fallback_models=ollama.get("fallback_models", ["mistral", "phi3"]),
        temperature=ollama.get("temperature", 0.7),
        max_tokens=ollama.get("max_tokens", 2048),
        context_window=ollama.get("context_window", 131072),
        top_p=ollama.get("top_p", 0.9),
        seed=ollama.get("seed"),
        short_max_tokens=ollama.get("short_max_tokens", 150),
        long_max_tokens=ollama.get("long_max_tokens", 16384),
        long_min_tokens=ollama.get("long_min_tokens", 0),
    )


def get_test_timeout(timeout_type: str = "default") -> int:
    """Get timeout for specific test type."""
    config_data = load_test_ollama_config()
    timeouts = config_data.get("ollama", {}).get("test_timeouts", {})
    return timeouts.get(timeout_type, timeouts.get("default", 10))



