"""Higher-level services that coordinate the scraper workflow."""

from .scraper import OpenTasksScraper, ScrapeResult

__all__ = ["OpenTasksScraper", "ScrapeResult"]
