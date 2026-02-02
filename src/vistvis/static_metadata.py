from pydantic import BaseModel
import yaml


class StaticMetadata(BaseModel):
    patch_size: int  # static
    num_special_tokens: int  # static

    def to_json(self) -> str:
        """Serialize the metadata to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "Metadata":
        """Deserialize the metadata from a JSON string."""
        return cls.model_validate_json(json_str)

    # load from yaml file define under path env varible: VIS_TVIS_METADATA_PATH
    @classmethod
    def from_yaml_file(cls, file_path: str) -> "StaticMetadata":
        """Deserialize the metadata from a YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
