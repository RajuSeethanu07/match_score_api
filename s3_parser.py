"""
s3_parser.py
Production-ready async S3 PDF extraction pipeline for Match Score API
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import anyio
import boto3
import fitz
import httpx
from botocore.config import Config

from config import settings

logger = logging.getLogger(__name__)


class S3Parser:
    """
    Handles:
    - Fresh signed URL generation
    - Expired URL recovery
    - PDF streaming
    - PDF text extraction
    """

    def __init__(self, timeout: int = settings.s3_timeout):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ═══════════════════════════════════════
    # HTTP CLIENT
    # ═══════════════════════════════════════

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("S3Parser HTTP client closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    # ═══════════════════════════════════════
    # SIGNED URL GENERATION
    # ═══════════════════════════════════════

    def get_signed_url(
        self,
        key_or_url: str,
        expires_in_seconds: int = 3600,
    ) -> str:
        """
        Converts:
        - plain S3 object key
        - expired S3 signed URL
        - raw S3 URL

        into a fresh downloadable signed URL.
        """

        if not key_or_url:
            logger.warning("Empty S3 key/url received")
            return ""

        try:
            bucket_name = settings.s3_bucket
            region_name = settings.aws_region

            access_key = settings.aws_access_key_id
            secret_key = settings.aws_secret_access_key

            object_key = ""

            # ---------------------------------------------------
            # HANDLE FULL URL INPUT
            # ---------------------------------------------------

            if key_or_url.startswith(("https://", "http://")):

                parsed_url = urlparse(key_or_url)

                object_key = parsed_url.path.lstrip("/")

                # Remove bucket prefix if present
                if object_key.startswith(f"{bucket_name}/"):
                    object_key = object_key[len(bucket_name) + 1:]

                logger.info(
                    "Extracted object key from URL => %s",
                    object_key,
                )

            else:
                object_key = key_or_url.strip()

                logger.info(
                    "Using raw S3 object key => %s",
                    object_key,
                )

            if not object_key:
                logger.error("Resolved object key is empty")
                return ""

            # ---------------------------------------------------
            # CREATE S3 CLIENT
            # ---------------------------------------------------

            s3_client = boto3.client(
                "s3",
                region_name=region_name,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version="s3v4"),
            )

            # ---------------------------------------------------
            # GENERATE FRESH SIGNED URL
            # ---------------------------------------------------

            signed_url = s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": bucket_name,
                    "Key": object_key,
                },
                ExpiresIn=expires_in_seconds,
            )

            logger.info(
                "Fresh signed URL generated successfully for key=%s",
                object_key,
            )

            return signed_url

        except Exception as e:
            logger.exception(
                "Signed URL generation failed: %s",
                str(e),
            )
            return ""

    # ═══════════════════════════════════════
    # PDF DOWNLOAD
    # ═══════════════════════════════════════

    async def fetch_pdf_stream(self, s3_url: str) -> BytesIO:
        """
        Download PDF into memory buffer.
        """

        if not s3_url:
            raise ValueError("S3 URL cannot be empty")

        logger.info("Fetching PDF stream")

        try:
            async with self.client.stream("GET", s3_url) as response:

                logger.info(
                    "S3 response status => %s",
                    response.status_code,
                )

                response.raise_for_status()

                pdf_data = await response.aread()

                if not pdf_data:
                    raise ValueError("Downloaded PDF is empty")

                logger.info(
                    "PDF downloaded successfully size=%d bytes",
                    len(pdf_data),
                )

                return BytesIO(pdf_data)

        except httpx.HTTPStatusError as e:

            logger.error(
                "HTTP error while downloading PDF status=%s error=%s",
                e.response.status_code,
                str(e),
            )

            raise

        except Exception as e:

            logger.exception(
                "Failed fetching PDF stream: %s",
                str(e),
            )

            raise

    # ═══════════════════════════════════════
    # PDF TEXT EXTRACTION
    # ═══════════════════════════════════════

    def _sync_extract_text(self, pdf_buffer: BytesIO) -> str:
        """
        CPU-heavy PDF extraction executed in worker thread.
        """

        pdf_buffer.seek(0)

        raw_pdf_data = pdf_buffer.read()

        logger.info(
            "Starting PDF extraction buffer_size=%d",
            len(raw_pdf_data),
        )

        if not raw_pdf_data:
            logger.warning("PDF buffer is empty")
            return ""

        pdf_document = None

        try:
            pdf_document = fitz.open(
                stream=raw_pdf_data,
                filetype="pdf",
            )

            logger.info(
                "PDF opened successfully pages=%d",
                pdf_document.page_count,
            )

            if pdf_document.page_count == 0:
                logger.warning("PDF contains zero pages")
                return ""

            pages = []

            for page_num in range(pdf_document.page_count):

                try:
                    page = pdf_document[page_num]

                    page_text = page.get_text("text")

                    logger.info(
                        "Page=%d extracted_chars=%d",
                        page_num,
                        len(page_text),
                    )

                    if page_text and page_text.strip():
                        pages.append(page_text.strip())

                except Exception as e:
                    logger.warning(
                        "Failed extracting page=%d error=%s",
                        page_num,
                        str(e),
                    )

            final_text = "\n\n".join(pages).strip()

            logger.info(
                "Final extracted text chars=%d",
                len(final_text),
            )

            return final_text

        except Exception as e:

            logger.exception(
                "PDF parsing failed: %s",
                str(e),
            )

            return ""

        finally:
            if pdf_document:
                pdf_document.close()

    async def extract_text_from_pdf(
        self,
        pdf_buffer: BytesIO,
    ) -> str:
        """
        Async-safe PDF extraction.
        """

        try:
            if pdf_buffer.getbuffer().nbytes == 0:
                logger.warning("PDF buffer has zero bytes")
                return ""

            extracted_text = await anyio.to_thread.run_sync(
                self._sync_extract_text,
                pdf_buffer,
            )

            logger.info(
                "PDF text extraction completed chars=%d",
                len(extracted_text),
            )

            return extracted_text.strip()

        except Exception as e:

            logger.exception(
                "PDF extraction failed: %s",
                str(e),
            )

            return ""

        finally:
            pdf_buffer.seek(0)

    # ═══════════════════════════════════════
    # COMPLETE PARSING PIPELINE
    # ═══════════════════════════════════════

    async def parse_s3_pdf(self, key_or_url: str) -> str:
        """
        Full S3 PDF parsing pipeline.
        """

        try:
            signed_url = self.get_signed_url(key_or_url)

            if not signed_url:
                logger.error("Failed generating signed URL")
                return ""

            logger.info("Generated signed URL successfully")

            pdf_buffer = await self.fetch_pdf_stream(signed_url)

            extracted_text = await self.extract_text_from_pdf(
                pdf_buffer
            )

            if not extracted_text:
                logger.warning("No text extracted from PDF")

            return extracted_text

        except Exception as e:

            logger.exception(
                "S3 PDF parsing pipeline failed: %s",
                str(e),
            )

            return ""

    # ═══════════════════════════════════════
    # JD PARSER
    # ═══════════════════════════════════════

    async def parse_jd_from_s3(
        self,
        jd_key_or_url: str,
    ) -> str:

        logger.info("Starting JD PDF extraction")

        return await self.parse_s3_pdf(jd_key_or_url)

    # ═══════════════════════════════════════
    # RESUME PARSER
    # ═══════════════════════════════════════

    async def parse_resume_from_s3(
        self,
        resume_key_or_url: str,
    ) -> str:

        logger.info("Starting Resume PDF extraction")

        return await self.parse_s3_pdf(resume_key_or_url)

    # ═══════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════

    async def extract_text(
        self,
        s3_url: str,
        label: str = "",
    ) -> str:
        """
        Unified extraction interface used by main.py
        """

        if not s3_url:
            logger.warning(
                "Empty S3 input received label=%s",
                label,
            )
            return ""

        try:

            normalized_label = label.lower().strip()

            if normalized_label == "jd":
                return await self.parse_jd_from_s3(s3_url)

            if normalized_label in ("cv", "resume"):
                return await self.parse_resume_from_s3(s3_url)

            return await self.parse_s3_pdf(s3_url)

        except Exception as e:

            logger.exception(
                "S3 extraction failed label=%s error=%s",
                label,
                str(e),
            )

            return ""


# Singleton instance
s3_parser = S3Parser()