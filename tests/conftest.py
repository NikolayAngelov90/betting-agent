"""Shared pytest fixtures and environment setup."""

import os

# Opt into the insecure default HMAC key for tests. Production runs require
# MODEL_HMAC_KEY to be set or this env var to be set explicitly — see
# src/models/ml_models.py for the rationale.
os.environ.setdefault("BETTING_AGENT_ALLOW_DEFAULT_HMAC", "1")
