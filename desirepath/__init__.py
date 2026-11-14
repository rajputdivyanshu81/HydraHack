"""DesirePath: a self-healing graph on HydraDB that repairs itself where users ask."""
import os

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:  # python-dotenv is optional; env vars still work
    pass
