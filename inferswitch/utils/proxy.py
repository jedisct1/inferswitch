"""
HTTP/HTTPS proxy utilities for InferSwitch.
"""

import asyncio
import logging
import socket
import threading
import urllib.parse
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ProxyHandler:
    """Handle HTTP/HTTPS proxy requests."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _read_http_response(self, socket: socket.socket) -> bytes:
        """Read HTTP response with proper parsing."""
        response_data = b''

        # Read headers in chunks instead of byte-by-byte
        headers_complete = False

        while not headers_complete:
            chunk = socket.recv(4096)
            if not chunk:
                break
            response_data += chunk

            if b'\r\n\r\n' in response_data:
                headers_complete = True
                break

        if not headers_complete:
            return response_data

        # Parse headers to determine content length
        headers_end = response_data.find(b'\r\n\r\n') + 4
        headers_part = response_data[:headers_end].decode('utf-8', errors='ignore')
        body_so_far = response_data[headers_end:]

        content_length = None
        is_chunked = False
        connection_close = False

        for line in headers_part.split('\r\n'):
            line_lower = line.lower()
            if line_lower.startswith('content-length:'):
                try:
                    content_length = int(line.split(':', 1)[1].strip())
                except:
                    pass
            elif (line_lower.startswith('transfer-encoding:') and
                  'chunked' in line_lower):
                is_chunked = True
            elif line_lower.startswith('connection:') and 'close' in line_lower:
                connection_close = True

        # Read the body based on the headers
        if is_chunked:
            # Handle chunked transfer encoding
            chunked_body = self._read_chunked_body(socket, body_so_far)
            response_data = response_data[:headers_end] + chunked_body
        elif content_length is not None:
            # Handle content-length
            remaining = content_length - len(body_so_far)
            while remaining > 0:
                chunk = socket.recv(min(remaining, 4096))
                if not chunk:
                    break
                response_data += chunk
                remaining -= len(chunk)
        elif connection_close:
            # Connection: close - read until connection closes
            socket.settimeout(5)  # Shorter timeout for connection close
            try:
                while True:
                    chunk = socket.recv(4096)
                    if not chunk:
                        break
                    response_data += chunk
            except (socket.timeout, socket.error):
                pass
            finally:
                socket.settimeout(self.timeout)
        else:
            # HTTP/1.1 default - read for a short time and assume complete
            socket.settimeout(2)
            try:
                while True:
                    chunk = socket.recv(4096)
                    if not chunk:
                        break
                    response_data += chunk
            except (socket.timeout, socket.error):
                pass
            finally:
                socket.settimeout(self.timeout)

        return response_data

    def _read_chunked_body(self, socket: socket.socket, initial_body: bytes) -> bytes:
        """Read chunked transfer encoding body."""
        body_data = initial_body

        try:
            while True:
                # Read chunk size line
                chunk_size_line = b''
                while not chunk_size_line.endswith(b'\r\n'):
                    byte = socket.recv(1)
                    if not byte:
                        return body_data
                    chunk_size_line += byte

                # Parse chunk size
                try:
                    chunk_size = int(chunk_size_line.strip().decode('utf-8'), 16)
                except:
                    break

                if chunk_size == 0:
                    # Read final \r\n and any trailing headers
                    socket.recv(2)  # \r\n
                    break

                # Read chunk data
                chunk_data = b''
                while len(chunk_data) < chunk_size:
                    remaining = chunk_size - len(chunk_data)
                    data = socket.recv(min(remaining, 4096))
                    if not data:
                        return body_data
                    chunk_data += data

                body_data += chunk_data

                # Read trailing \r\n
                socket.recv(2)

        except Exception as e:
            logger.debug(f"Error reading chunked body: {e}")

        return body_data

    async def handle_http_request(
        self, request_data: bytes, method: str, url: str
    ) -> bytes:
        """Handle HTTP requests (GET, POST, etc.)."""
        try:
            # Parse URL
            parsed_url = urllib.parse.urlparse(url)

            # If URL doesn't have scheme, it's a relative URL
            # (shouldn't happen in proxy)
            if not parsed_url.scheme:
                logger.warning(f"Relative URL in proxy request: {url}")
                return b'HTTP/1.1 400 Bad Request\r\n\r\n'

            # Create request to target server
            target_host = parsed_url.hostname
            target_port = (
                parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
            )

            # Create socket connection to target
            target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_socket.settimeout(self.timeout)
            target_socket.connect((target_host, target_port))

            # Modify request to remove absolute URL (make it relative)
            request_str = request_data.decode('utf-8', errors='ignore')
            lines = request_str.split('\n')

            # Reconstruct path from parsed URL
            path = parsed_url.path
            if parsed_url.query:
                path += '?' + parsed_url.query
            if parsed_url.fragment:
                path += '#' + parsed_url.fragment

            if not path:
                path = '/'

            # Replace first line with relative path
            lines[0] = f"{method} {path} HTTP/1.1"

            # Send modified request to target
            modified_request = '\n'.join(lines).encode('utf-8')
            target_socket.send(modified_request)

            # Collect response
            response_data = b''
            while True:
                chunk = target_socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk

            target_socket.close()
            return response_data

        except Exception as e:
            logger.error(f"Error in HTTP request: {e}")
            return b'HTTP/1.1 502 Bad Gateway\r\n\r\n'

    def handle_https_tunnel(self, client_socket: socket.socket, url: str) -> None:
        """Handle HTTPS CONNECT requests for SSL tunneling."""
        try:
            # Parse host and port from URL
            if ':' in url:
                host, port = url.split(':', 1)
                port = int(port)
            else:
                host = url
                port = 443

            # Create connection to target server
            target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_socket.settimeout(self.timeout)
            target_socket.connect((host, port))

            # Send 200 Connection established response
            client_socket.send(b'HTTP/1.1 200 Connection established\r\n\r\n')

            # Start bidirectional data relay
            def relay_data(source: socket.socket, destination: socket.socket):
                try:
                    while True:
                        data = source.recv(4096)
                        if not data:
                            break
                        destination.send(data)
                except Exception as e:
                    logger.debug(f"Relay error: {e}")
                finally:
                    try:
                        source.close()
                    except:
                        pass
                    try:
                        destination.close()
                    except:
                        pass

            # Start relay threads
            client_to_server = threading.Thread(
                target=relay_data,
                args=(client_socket, target_socket)
            )
            server_to_client = threading.Thread(
                target=relay_data,
                args=(target_socket, client_socket)
            )

            client_to_server.daemon = True
            server_to_client.daemon = True

            client_to_server.start()
            server_to_client.start()

            # Wait for threads to complete
            client_to_server.join()
            server_to_client.join()

        except Exception as e:
            logger.error(f"Error in HTTPS tunnel: {e}")
            # Send error response
            try:
                client_socket.send(b'HTTP/1.1 502 Bad Gateway\r\n\r\n')
            except:
                pass


class ProxyServer:
    """HTTP/HTTPS proxy server."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 1236, timeout: int = 30
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.server_socket: Optional[socket.socket] = None
        self.handler = ProxyHandler(timeout)
        self.running = False

    async def start(self):
        """Start the proxy server."""
        if self.running:
            return

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            logger.info(f"HTTP/HTTPS proxy server listening on {self.host}:{self.port}")

            # Run server in background thread (don't await)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self._run_server)

        except Exception as e:
            logger.error(f"Error starting proxy server: {e}")
            self.running = False
            if self.server_socket:
                self.server_socket.close()

    def _run_server(self):
        """Run the server loop (blocking)."""
        while self.running:
            try:
                if not self.server_socket:
                    break

                client_socket, client_address = self.server_socket.accept()
                logger.debug(f"Proxy connection from {client_address}")

                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()

            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting proxy connection: {e}")
                break

    def _handle_http_request_sync(
        self, request_data: bytes, method: str, url: str
    ) -> bytes:
        """Handle HTTP requests synchronously."""
        try:
            # Parse URL
            parsed_url = urllib.parse.urlparse(url)

            # If URL doesn't have scheme, it's a relative URL
            # (shouldn't happen in proxy)
            if not parsed_url.scheme:
                logger.warning(f"Relative URL in proxy request: {url}")
                return b'HTTP/1.1 400 Bad Request\r\n\r\n'

            # Create request to target server
            target_host = parsed_url.hostname
            target_port = (
                parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
            )

            logger.debug(f"Connecting to {target_host}:{target_port}")

            # Create socket connection to target
            target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_socket.settimeout(self.timeout)
            target_socket.connect((target_host, target_port))

            # Modify request to remove absolute URL (make it relative)
            request_str = request_data.decode('utf-8', errors='ignore')
            lines = request_str.replace('\r\n', '\n').split('\n')

            # Reconstruct path from parsed URL
            path = parsed_url.path
            if parsed_url.query:
                path += '?' + parsed_url.query
            if parsed_url.fragment:
                path += '#' + parsed_url.fragment

            if not path:
                path = '/'

            # Replace first line with relative path
            lines[0] = f"{method} {path} HTTP/1.1"

            # Send modified request to target - fix line endings
            modified_request = '\r\n'.join(lines).encode('utf-8')
            logger.debug(f"Sending request: {modified_request[:200]}...")
            target_socket.send(modified_request)

            # Read response with proper HTTP parsing
            response_data = self._read_http_response(target_socket)

            target_socket.close()
            logger.debug(f"Received response: {len(response_data)} bytes")
            return response_data

        except Exception as e:
            logger.error(f"Error in HTTP request to {url}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return b'HTTP/1.1 502 Bad Gateway\r\n\r\n'

    def _handle_client(
        self, client_socket: socket.socket, client_address: Tuple[str, int]
    ):
        """Handle a client connection."""
        try:
            client_socket.settimeout(self.timeout)

            # Read the request
            request_data = client_socket.recv(4096)
            if not request_data:
                return

            request_str = request_data.decode('utf-8', errors='ignore')
            request_lines = request_str.split('\n')

            if not request_lines:
                return

            # Parse the first line (method, URL, HTTP version)
            first_line = request_lines[0].strip()
            parts = first_line.split(' ')

            if len(parts) < 3:
                logger.warning(
                    f"Invalid proxy request from {client_address}: {first_line}"
                )
                return

            method = parts[0]
            url = parts[1]
            http_version = parts[2]

            logger.info(f"Proxy {client_address} - {method} {url}")

            if method == 'CONNECT':
                # Handle HTTPS tunneling
                self.handler.handle_https_tunnel(client_socket, url)
            else:
                # Handle HTTP requests synchronously
                response = self._handle_http_request_sync(request_data, method, url)
                client_socket.send(response)

        except Exception as e:
            logger.error(f"Error handling proxy client {client_address}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass

    async def stop(self):
        """Stop the proxy server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        logger.info("HTTP/HTTPS proxy server stopped")


# Global proxy server instance
_proxy_server: Optional[ProxyServer] = None


async def start_proxy_server(
    host: str = "127.0.0.1", port: int = 1236, timeout: int = 30
):
    """Start the global proxy server."""
    global _proxy_server

    if _proxy_server and _proxy_server.running:
        return

    _proxy_server = ProxyServer(host, port, timeout)

    # Start in background task
    asyncio.create_task(_proxy_server.start())


async def stop_proxy_server():
    """Stop the global proxy server."""
    global _proxy_server

    if _proxy_server:
        await _proxy_server.stop()
        _proxy_server = None


def get_proxy_server() -> Optional[ProxyServer]:
    """Get the global proxy server instance."""
    return _proxy_server
