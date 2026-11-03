"""Production WSGI entry point, usable on Windows and in Linux containers."""
import os
import subprocess
from local_model import start_optional_model, stop_owned_model


def main():
    process = None
    try:
        process = start_optional_model()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Optional local model setup failed ({type(error).__name__}); OCR and retrieval remain available.", flush=True)
    try:
        from app import create_app
        from waitress import serve
        host = os.environ.get("APP_HOST", "127.0.0.1")
        port = int(os.environ.get("PORT", "5003"))
        app = create_app()
        print(f"INKLAB 3.0: http://{host}:{port}/ (Waitress)", flush=True)
        serve(app, host=host, port=port, threads=4, channel_timeout=90,
              max_request_body_size=3 * 1024 * 1024, clear_untrusted_proxy_headers=True,
              trusted_proxy=os.environ.get("INKLAB_TRUSTED_PROXY", "127.0.0.1"),
              trusted_proxy_headers={"x-forwarded-proto"})
    finally:
        stop_owned_model(process)


if __name__ == "__main__":
    main()

