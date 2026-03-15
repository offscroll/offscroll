"""Email digest renderer."""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from jinja2 import Template

from offscroll.models import CuratedEdition

logger = logging.getLogger(__name__)


def _load_edition(
    config: dict,
    edition_path: Path | None = None,
    edition: CuratedEdition | None = None,
) -> CuratedEdition:
    """Load a CuratedEdition from a path or use the provided one.

    If neither is provided, look for the latest edition in the
    data directory.
    """
    if edition is not None:
        return edition
    if edition_path is not None:
        return CuratedEdition.from_json(edition_path)
    data_dir = Path(config["output"]["data_dir"]).expanduser()
    editions = sorted(data_dir.glob("edition-*.json"), reverse=True)
    if not editions:
        raise FileNotFoundError(
            f"No edition files found in {data_dir}. Run 'offscroll curate' first."
        )
    return CuratedEdition.from_json(editions[0])


def render_digest(
    config: dict,
    edition_path: Path | None = None,
    edition: CuratedEdition | None = None,
    send: bool = False,
) -> Path:
    """Render a CuratedEdition to an HTML email digest.

    Produces a simple sequential HTML email: each section header
    followed by its items as text blocks with author and content.
    No columns, no page breaks, no complex CSS.

    Args:
        config: The OffScroll config dict.
        edition_path: Path to edition.json. If None, uses the latest.
        edition: Pre-loaded CuratedEdition. If provided, edition_path
                 is ignored.
        send: If True, send the email via SMTP after rendering.

    Returns:
        Path to the generated HTML file.
    """
    ed = _load_edition(config, edition_path, edition)

    # Inline HTML template for email digest
    template_str = """<html>
<head><style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 600px;
    margin: 0 auto;
    padding: 20px;
}
h1 {
    font-size: 28px;
    margin: 0 0 10px 0;
    font-weight: bold;
}
.edition-meta {
    color: #666;
    font-size: 14px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ddd;
}
.edition-note {
    font-style: italic;
    color: #555;
    margin-bottom: 20px;
    padding: 10px;
    background-color: #f5f5f5;
    border-left: 3px solid #ccc;
}
h2 {
    font-size: 18px;
    margin: 30px 0 15px 0;
    font-weight: bold;
    padding-bottom: 8px;
    border-bottom: 2px solid #333;
}
.item {
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid #eee;
}
.item h3 {
    margin: 0 0 8px 0;
    font-size: 16px;
    font-weight: bold;
}
.item p {
    margin: 0 0 8px 0;
}
.meta {
    font-size: 12px;
    color: #999;
    margin-top: 8px;
}
blockquote {
    margin: 20px 0;
    padding: 15px 20px;
    border-left: 4px solid #007acc;
    background-color: #f5f8ff;
    font-style: italic;
    color: #555;
}
blockquote br {
    display: none;
}
blockquote::after {
    content: "";
}
</style></head>
<body>
    <h1>{{ edition.title }}</h1>
    <div class="edition-meta">
        <strong>{{ edition.subtitle }}</strong> &mdash; {{ edition.date }}
    </div>
    {% if edition.editorial_note %}
    <div class="edition-note">{{ edition.editorial_note }}</div>
    {% endif %}
    {% for section in sections %}
    <h2>{{ section.heading }}</h2>
    {% for item in section.items %}
        {% if item.thread_id is defined %}
        {# CuratedThread #}
        <div class="item">
            <h3>{{ item.headline }}</h3>
            <p><strong>{{ item.author }}</strong></p>
            {% if item.editorial_note %}
            <p><em>{{ item.editorial_note }}</em></p>
            {% endif %}
            {% for sub in item.items %}
            <p>{{ sub.display_text }}</p>
            {% endfor %}
        </div>
        {% else %}
        {# CuratedItem #}
        <div class="item">
            {% if item.title %}<h3>{{ item.title }}</h3>{% endif %}
            <p>{{ item.display_text }}</p>
            <p class="meta">— {{ item.author }}</p>
            {% if item.editorial_note %}
            <p><em>{{ item.editorial_note }}</em></p>
            {% endif %}
        </div>
        {% endif %}
    {% endfor %}
    {% endfor %}
    {% if pull_quotes %}
    <h2>Notable Quotes</h2>
    {% for quote in pull_quotes %}
    <blockquote>
        "{{ quote.text }}"<br>
        <strong>— {{ quote.attribution }}</strong>
    </blockquote>
    {% endfor %}
    {% endif %}
</body>
</html>"""

    # Render HTML
    template = Template(template_str)
    html_content = template.render(
        edition=ed.edition,
        sections=ed.sections,
        pull_quotes=ed.pull_quotes,
    )

    # Write to disk
    output_dir = Path(config["output"]["data_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"digest-{ed.edition.date}.html"
    output_path.write_text(html_content)

    logger.info("Email digest written to %s", output_path)

    # Send via SMTP if requested
    if send:
        _send_digest_email(config, ed, html_content)

    return output_path


def _send_digest_email(config: dict, edition: CuratedEdition, html_content: str) -> None:
    """Send the email digest via SMTP.

    If SMTP config is incomplete or send fails, log the error but
    do not raise (the HTML file is still useful).
    """
    email_config = config.get("email", {})

    # Validate required fields
    smtp_host = email_config.get("smtp_host", "").strip()
    smtp_port = email_config.get("smtp_port", 587)
    from_address = email_config.get("from_address", "").strip()
    to_addresses = email_config.get("to_addresses", [])

    if not smtp_host or not from_address or not to_addresses:
        logger.error(
            "SMTP config incomplete: host=%s, from=%s, to_count=%s",
            smtp_host,
            from_address,
            len(to_addresses),
        )
        raise ValueError(
            "Email configuration incomplete: smtp_host, from_address, and to_addresses must be set."
        )

    # Get credentials from environment
    smtp_user_env = email_config.get("smtp_user_env", "OFFSCROLL_SMTP_USER")
    smtp_password_env = email_config.get("smtp_password_env", "OFFSCROLL_SMTP_PASSWORD")

    try:
        smtp_user = os.environ.get(smtp_user_env)
        smtp_password = os.environ.get(smtp_password_env)

        if not smtp_user or not smtp_password:
            logger.error(
                "SMTP credentials not found in environment: user_env=%s, pass_env=%s",
                smtp_user_env,
                smtp_password_env,
            )
            raise ValueError(
                f"SMTP credentials not found in environment variables: "
                f"{smtp_user_env}, {smtp_password_env}"
            )

        # Compose message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{edition.edition.title} - {edition.edition.date}"
        msg["From"] = from_address
        msg["To"] = ", ".join(to_addresses)
        msg.attach(MIMEText(html_content, "html"))

        # Send
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        logger.info("Email digest sent to %s recipients", len(to_addresses))

    except Exception as e:
        logger.error("Failed to send email digest: %s", str(e))
        # Do not raise -- the HTML file was already saved
