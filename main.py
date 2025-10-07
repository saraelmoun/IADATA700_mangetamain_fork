"""Main entrypoint to run the Streamlit app programmatically.

Typical usage (still prefer direct streamlit run for hot reload):
    python -m streamlit run src/app.py
or
    python main.py  (will emit guidance)
"""

from src import (
    App,
)


def run():
    # This allows alternative orchestration if needed later.
    App().run()


if (
    __name__
    == "__main__"
):
    # Provide a hint if user executes python main.py directly
    try:
        import streamlit.web.bootstrap as bootstrap  # type: ignore

        bootstrap.run(
            "src/app.py",
            "streamlit run",
            [],
            {},
        )
    except Exception:
        # Fallback: direct run (limited features vs streamlit CLI)
        run()
