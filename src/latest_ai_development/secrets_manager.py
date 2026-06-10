from __future__ import annotations

import base64
import json
import os
from functools import lru_cache

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _resolve_region(region_name: str | None = None) -> str:
    """
    Resolve AWS region from explicit argument or environment.
    """
    return (
        region_name
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def get_secret(secret_name: str, region_name: str | None = None) -> str:
    """
    Retrieve raw secret string from AWS Secrets Manager.
    """

    if not secret_name:
        return ""

    region = _resolve_region(region_name)

    client = boto3.client("secretsmanager", region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to retrieve secret '{secret_name}': {exc}") from exc

    if response.get("SecretString"):
        return response["SecretString"]

    if response.get("SecretBinary"):
        return base64.b64decode(response["SecretBinary"]).decode("utf-8")

    return ""


def _extract_value_from_secret(secret_value: str, preferred_key: str) -> str:
    """
    Extract a specific key from JSON secrets.
    """

    if not secret_value:
        return ""

    try:
        parsed = json.loads(secret_value)
    except json.JSONDecodeError:
        # Secret stored as plain string
        return secret_value.strip()

    if isinstance(parsed, dict):

        if preferred_key in parsed:
            return str(parsed[preferred_key]).strip()

        # fallback keys
        for key in ("OPENAI_API_KEY", "api_key", "key", "value"):
            if key in parsed:
                return str(parsed[key]).strip()

    return ""


@lru_cache(maxsize=8)
def resolve_secret_or_env(
    env_var_name: str,
    secret_name_env_var: str | None = None,
    preferred_secret_key: str | None = None,
) -> str:
    """
    Resolve a value using the following priority:

    1. Environment variable
    2. AWS Secrets Manager
    """

    # 1️⃣ direct environment variable
    direct_value = os.getenv(env_var_name, "").strip()
    if direct_value:
        return direct_value

    # 2️⃣ secret lookup
    if not secret_name_env_var:
        return ""

    secret_name = os.getenv(secret_name_env_var, "").strip()
    if not secret_name:
        return ""

    secret_value = get_secret(secret_name)

    return _extract_value_from_secret(
        secret_value,
        preferred_secret_key or env_var_name,
    )


def resolve_openai_api_key() -> str:
    """
    Resolve OpenAI API key from ENV or AWS Secrets Manager.
    """
    return resolve_secret_or_env(
        env_var_name="OPENAI_API_KEY",
        secret_name_env_var="OPENAI_API_KEY_SECRET",
        preferred_secret_key="OPENAI_API_KEY",
    )

def main():
    """
    Simple manual test for secrets resolution.
    Run this file directly to verify environment variables
    and AWS Secrets Manager integration.
    """

    print("=== Secrets Manager Test ===")

    print("AWS_REGION:", os.getenv("AWS_REGION"))
    print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    print("OPENAI_API_KEY_SECRET:", os.getenv("OPENAI_API_KEY_SECRET"))

    print("\n--- Testing resolve_openai_api_key() ---")

    try:
        key = resolve_openai_api_key()

        if key:
            print("SUCCESS: OpenAI API key resolved")
            print("Key prefix:", key[:10], "...")
        else:
            print("WARNING: No key resolved")

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()