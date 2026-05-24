"""
PolyBotKing Health Dashboard
=============================
Simple HTTP health/status endpoint for monitoring.
Runs on port 8080 for Docker health checks and monitoring.
"""

import asyncio
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from polybotking.config import settings
from polybotking.logger import get_logger

logger = get_logger("dashboard")


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health and status endpoints."""

    def do_GET(self):
        if self.path == "/health":
            self._respond_health()
        elif self.path == "/status":
            self._respond_status()
        elif self.path == "/metrics":
            self._respond_metrics()
        else:
            self.send_response(404)
            self.end_headers()

    def _respond_health(self):
        """Simple health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        }).encode())

    def _respond_status(self):
        """Detailed status endpoint."""
        try:
            status = self._get_bot_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _respond_metrics(self):
        """Prometheus-compatible metrics."""
        try:
            metrics = self._get_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(metrics.encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            try:
                self.wfile.write(f"# error: {e}\n".encode())
            except Exception:
                pass

    def _get_bot_status(self) -> dict:
        """Get current bot status."""
        return {
            "bot": "PolyBotKing",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "bankroll": settings.risk.initial_bankroll,
                "kelly_fraction": settings.risk.kelly_fraction,
                "max_positions": settings.trading.max_concurrent_positions,
                "scan_interval_s": settings.trading.scan_interval_seconds,
            }
        }

    def _get_metrics(self) -> str:
        """Generate Prometheus-format metrics."""
        lines = [
            "# HELP polybotking_up Bot is running",
            "# TYPE polybotking_up gauge",
            "polybotking_up 1",
            "",
            f"# HELP polybotking_bankroll Current bankroll in USD",
            f"# TYPE polybotking_bankroll gauge",
            f"polybotking_bankroll {settings.risk.initial_bankroll}",
        ]
        return "\n".join(lines)

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def run_dashboard(port: int = 8080):
    """Start the health dashboard HTTP server."""
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info("dashboard_started", port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("dashboard_interrupted")
    finally:
        try:
            server.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    run_dashboard()
