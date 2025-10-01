from typing import Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def hyphenize(field: str):
    return field.replace("_", "-")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix="TAGGER_",
        alias_generator=hyphenize,
        env_file=(".tag.env"),
    )

    model_repo: str = Field(
        default="SmilingWolf/wd-eva02-large-tagger-v3",
        description="HuggingFace repository for the TIMM model.",
    )
    tags_filename: str = Field(
        default="selected_tags.csv", description="CSV file with tag names in the repo."
    )
    thresholds_filename: str = Field(
        default="thresholds.csv",
        description="Optional CSV for category-level thresholds.",
    )
    host: str = Field(default="0.0.0.0", description="Server host.")
    port: int = Field(default=8000, description="Server port.")
    device: str = Field(
        default="cuda",
        description="Base device type. 'cuda' will spawn workers across all available GPUs.",
    )
    # --- NEW: Control number of GPU workers ---
    device_ids: Optional[List[int]] = Field(
        default=None,
        description="Comma-separated list of GPU device IDs to use (e.g., '0,1,3'). "
        "If not set, uses all available GPUs. "
        "Provide an empty value (e.g., --device-ids=) for CPU-only.",
    )
    compile_model: bool = Field(
        default=False, description="Enable torch.compile() for the model."
    )
    use_recall: bool = Field(
        default=False, description="Use Recall instead of best_threshold"
    )
    log_level: str = Field(default="INFO", description="Logging level.")
    token: Optional[str] = Field(default=None, description="HuggingFace token")

    @field_validator("device_ids", mode="before")
    @classmethod
    def parse_device_ids(cls, v: Any) -> Any:
        """
        Handles comma-separated strings from CLI/env for List[int] fields.
        Crucially, it transforms an empty input ('--device-ids=') into a valid
        empty list for CPU-only mode. pydantic-settings pre-parses this
        empty string into [''], which this validator corrects to [].
        """
        # This is the key change: check for the pre-parsed list with an empty string.
        if v == ['']:
            return []
        
        # Also handle the raw empty string for other input sources.
        if v == '':
            return []

        # Let Pydantic's powerful parser handle all other valid cases:
        # - None -> None (when the argument is not provided at all)
        # - "0,1" -> ['0', '1'] -> [0, 1] (if not pre-split)
        # - ['0', '1'] -> [0, 1] (if pre-split)
        return v
