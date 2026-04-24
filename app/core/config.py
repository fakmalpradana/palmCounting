# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env.local is loaded after .env and takes precedence — use it for local
    # dev overrides (DEV_MODE, FRONTEND_URL, etc.) without touching production .env.
    model_config = SettingsConfigDict(
        env_file=[".env", ".env.local"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Set DEV_MODE=true in .env to bypass Google OAuth and Firestore entirely.
    # A local superadmin mock user is used instead — never enable in production.
    dev_mode: bool = False

    google_client_id: str = "dev"
    google_client_secret: str = "dev"
    jwt_secret: str = "dev-secret-change-me"
    jwt_access_expire_minutes: int = 15
    jwt_refresh_expire_days: int = 7
    firestore_project_id: str = ""
    frontend_url: str = "http://localhost:8000"
    gpu_worker_url: str = ""
    gcs_bucket_name: str = ""
    cleanup_max_age_hours: int = 24
    cleanup_secret: str = ""

    # SMTP settings for email verification and password reset
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    smtp_use_tls: bool = True


settings = Settings()
