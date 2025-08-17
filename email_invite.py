import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import BackgroundTasks

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "musaibaltafurahman@gmail.com"  # replace with your gmail
SMTP_PASSWORD = "cpmjabjxaywchgpq"


def send_invite_email(to_email: str, token: str):
    join_url = f"https://localhost:8080/join?token={token}"

    subject = "You're invited to join a Space!"
    body = f"""
    Hi,

    You’ve been invited to join a Space.  
    Click the link below to accept the invitation:

    {join_url}

    This invite will expire in 7 days.

    Thanks,
    The Team
    """

    # Build email
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Send
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
    except Exception as e:
        print(f"Error sending invite email to {to_email}: {e}")
