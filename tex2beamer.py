import sys
import logging
import argparse
import os
from core import generate_slides
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Generate Beamer slides from LaTeX papers."
    )
    parser.add_argument(
        "--arxiv_id",
        type=str,
        required=True,
        help="The arXiv ID of the paper to process",
    )
    parser.add_argument(
        "--use_linter", action="store_true", help="Whether to use the linter"
    )
    parser.add_argument(
        "--use_pdfcrop", action="store_true", help="Whether to use pdfcrop"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (optional; falls back to OPENAI_API_KEY env)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: from DEFAULT_MODEL env or gpt-4.1-2025-04-14)",
    )
    args = parser.parse_args()

    if not generate_slides(
        args.arxiv_id,
        args.use_linter,
        args.use_pdfcrop,
        api_key=args.api_key,
        model_name=args.model,
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
