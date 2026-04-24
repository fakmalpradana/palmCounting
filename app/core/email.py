# app/core/email.py
from __future__ import annotations

import logging
import smtplib
from datetime import timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.auth import create_email_token
from app.core.config import settings

log = logging.getLogger(__name__)


def _send(to: str, subject: str, html_body: str) -> None:
    """Send an email via SMTP. No-op if SMTP is not configured."""
    if not settings.smtp_host:
        log.warning("SMTP not configured — skipping email to %s", to)
        return

    msg = MIMEMultipart("alternative")
    msg["From"] = settings.smtp_from_email or settings.smtp_user
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    if settings.smtp_use_tls:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(msg["From"], [to], msg.as_string())
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(msg["From"], [to], msg.as_string())

    log.info("Email sent to %s: %s", to, subject)


def send_verification_email(to_email: str, user_uid: str) -> None:
    token = create_email_token(
        user_uid, token_type="email_verify", expires_delta=timedelta(hours=24)
    )
    link = f"{settings.frontend_url}?verify_token={token}"
    html = (
        f"<h2>Verify your email</h2>"
        f"<p>Click the link below to verify your email address:</p>"
        f'<p><a href="{link}">Verify Email</a></p>'
        f"<p>This link expires in 24 hours.</p>"
    )
    _send(to_email, "Verify your email — Palm Counter", html)


def send_password_reset_email(to_email: str, user_uid: str) -> None:
    token = create_email_token(
        user_uid, token_type="password_reset", expires_delta=timedelta(hours=1)
    )
    link = f"{settings.frontend_url}?reset_token={token}"
    html = (
        f"<h2>Reset your password</h2>"
        f"<p>Click the link below to reset your password:</p>"
        f'<p><a href="{link}">Reset Password</a></p>'
        f"<p>This link expires in 1 hour. If you did not request this, ignore this email.</p>"
    )
    _send(to_email, "Password reset — Palm Counter", html)
