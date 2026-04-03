# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_client_id: str
    google_client_secret: str
    jwt_secret: str
    jwt_access_expire_minutes: int = 15
    jwt_refresh_expire_days: int = 7
    firestore_project_id: str = ""
    frontend_url: str = "http://localhost:8080"
    gpu_worker_url: str = ""
    gcs_bucket_name: str = ""
    cleanup_max_age_hours: int = 24
    cleanup_secret: str = ""


settings = Settings()
