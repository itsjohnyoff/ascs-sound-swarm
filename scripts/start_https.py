#!/usr/bin/env python3
"""Serve the demo on a single HTTPS origin for phones and the dashboard."""

import argparse
import ipaddress
import os
import re
import shutil
import signal
import socket
import ssl
import subprocess
import sys
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Thread
from typing import TextIO

import segno
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


ROOT = Path(__file__).resolve().parents[1]
LAN_PORT = 8443
TUNNEL_PORT = 8000
TUNNEL_START_TIMEOUT = 30
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com(?![\w.-])")


def lan_ipv4() -> str:
    try:
        # Connecting a UDP socket selects a source interface without sending data.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            address = probe.getsockname()[0]
            ip = ipaddress.ip_address(address)
            if not (ip.is_loopback or ip.is_unspecified):
                return address
    except OSError:
        pass

    try:
        addresses = socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM
        )
    except OSError:
        addresses = []
    for _, _, _, _, (address, _) in addresses:
        ip = ipaddress.ip_address(address)
        if not (ip.is_loopback or ip.is_unspecified):
            return address
    raise RuntimeError("No LAN IPv4 address found. Connect to a local network and retry.")


def certificate_is_usable(cert_path: Path, key_path: Path, address: str) -> bool:
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        now = datetime.now(UTC)
        if not cert.not_valid_before_utc <= now < cert.not_valid_after_utc:
            return False
        if ipaddress.ip_address(address) not in san.get_values_for_type(x509.IPAddress):
            return False
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert_path, key_path)
    except (OSError, ValueError, x509.ExtensionNotFound):
        return False
    return True


def write_self_signed_certificate(cert_path: Path, key_path: Path, address: str) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ASCS Sound Swarm")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.IPAddress(ipaddress.ip_address(address)),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.DNSName("localhost"),
                ]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def ensure_certificate(address: str, directory: Path) -> tuple[Path, Path]:
    cert_path = directory / "lan.crt"
    key_path = directory / "lan.key"
    if certificate_is_usable(cert_path, key_path, address):
        return cert_path, key_path

    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as temporary:
        new_cert = Path(temporary) / cert_path.name
        new_key = Path(temporary) / key_path.name
        mkcert = shutil.which("mkcert")
        if mkcert:
            try:
                subprocess.run(
                    [
                        mkcert,
                        "-cert-file", str(new_cert),
                        "-key-file", str(new_key),
                        address, "localhost", "127.0.0.1",
                    ],
                    check=True,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"mkcert failed: {exc}", file=sys.stderr)
        if not mkcert or not certificate_is_usable(new_cert, new_key, address):
            write_self_signed_certificate(new_cert, new_key, address)
        new_key.chmod(0o600)
        new_key.replace(key_path)
        new_cert.replace(cert_path)
    return cert_path, key_path


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


ARP_LINE = re.compile(r"\((\d+\.\d+\.\d+\.\d+)\) at ([0-9a-f:]+)", re.I)


def arp_neighbors(arp_output: str, own_address: str) -> list[str]:
    """Other devices this laptop has recently seen on its own /24 (from `arp -an`)."""
    own = ipaddress.ip_address(own_address)
    subnet = ipaddress.ip_network(f"{own_address}/24", strict=False)
    found = []
    for address, mac in ARP_LINE.findall(arp_output):
        ip = ipaddress.ip_address(address)
        if ip == own or ip not in subnet or ip in (subnet.network_address, subnet.broadcast_address):
            continue
        if mac.lower() in ("ff:ff:ff:ff:ff:ff",) or ip.is_multicast:
            continue
        found.append(address)
    return sorted(set(found), key=ipaddress.ip_address)


def isolation_warning(neighbors: list[str], gateway: str | None, reachable) -> str | None:
    """Warn when other devices are on this Wi-Fi but none of them answer the laptop.

    Venue and campus Wi-Fi often isolates clients: phones reach the internet but not the laptop.
    The router always answers, so it doesn't count. Some phones ignore pings, so this is a hint.
    """
    peers = [ip for ip in neighbors if ip != gateway][:8]
    if not peers:
        return None
    if any(reachable(ip) for ip in peers):
        return None
    return ("Warning: no other device on this Wi-Fi answers the laptop. The network may block phones from "
            "reaching it (client isolation). If phones can't open the join URL, connect the laptop and phones "
            "to a phone hotspot and restart, or use --mode tunnel.")


def _ping(ip: str) -> bool:
    flag = ["-t", "1"] if sys.platform == "darwin" else ["-W", "1"]
    try:
        return subprocess.run(["ping", "-c", "1", *flag, ip], capture_output=True, timeout=3).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _gateway() -> str | None:
    try:
        out = subprocess.run(["route", "-n", "get", "default"] if sys.platform == "darwin" else ["ip", "route"],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"(?:gateway:|default via)\s+(\d+\.\d+\.\d+\.\d+)", out)
    return match.group(1) if match else None


def check_network(address: str) -> None:
    """Run in the background so the server starts immediately; print only if there's a problem."""
    try:
        arp = subprocess.run(["arp", "-an"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return
    from concurrent.futures import ThreadPoolExecutor

    neighbors = arp_neighbors(arp, address)
    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = dict(zip(neighbors, pool.map(_ping, neighbors)))
    warning = isolation_warning(neighbors, _gateway(), lambda ip: replies.get(ip, False))
    if warning:
        print(warning, file=sys.stderr, flush=True)


def serve(
    host: str,
    port: int,
    base_url: str,
    tls_args: tuple[str, ...] = (),
    tunnel: subprocess.Popen | None = None,
) -> int:
    command = [
        sys.executable, "-m", "uvicorn", "backend.server.app:app",
        "--host", host, "--port", str(port),
        *tls_args,
    ]
    server = subprocess.Popen(
        command, cwd=ROOT, env={**os.environ, "PUBLIC_BASE_URL": base_url}
    )
    try:
        join_url = f"{base_url}/phone.html"
        print(f"Phone join URL: {join_url}", flush=True)
        if sys.stdout.isatty():
            segno.make(join_url, micro=False).terminal(out=sys.stdout, compact=True)
        if tunnel is None:
            print("Phones need to trust the local TLS certificate or its issuer.", flush=True)
            return server.wait()
        while True:
            try:
                return server.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                status = tunnel.poll()
                if status is not None:
                    raise RuntimeError(f"cloudflared exited with status {status}.")
    finally:
        stop(server)


def read_tunnel_url(output: TextIO, result: Queue[tuple[str | None, str]]) -> None:
    # Keep draining the pipe after publishing the URL so cloudflared cannot block.
    recent_lines: deque[str] = deque(maxlen=5)
    reported = False
    with output:
        for line in output:
            if not reported:
                match = TUNNEL_URL.search(line)
                if match:
                    result.put((match.group(), ""))
                    reported = True
                else:
                    recent_lines.append(line.rstrip())
    if not reported:
        result.put((None, "\n".join(recent_lines)))


def serve_tunnel() -> int:
    cloudflared = shutil.which("cloudflared")
    if cloudflared is None:
        raise RuntimeError("Install cloudflared and make it available on PATH for tunnel mode.")

    # The quick tunnel can allocate its URL before the local origin starts.
    tunnel = subprocess.Popen(
        [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{TUNNEL_PORT}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    reader = None
    try:
        result: Queue[tuple[str | None, str]] = Queue()
        assert tunnel.stdout is not None
        reader = Thread(target=read_tunnel_url, args=(tunnel.stdout, result), daemon=True)
        reader.start()
        try:
            base_url, logs = result.get(timeout=TUNNEL_START_TIMEOUT)
        except Empty:
            raise RuntimeError("Timed out waiting for the cloudflared tunnel URL.") from None
        if base_url is None:
            message = "cloudflared stopped before reporting a tunnel URL."
            raise RuntimeError(f"{message}\n{logs}".rstrip())
        if tunnel.poll() is not None:
            raise RuntimeError("cloudflared stopped after reporting the tunnel URL.")
        return serve("127.0.0.1", TUNNEL_PORT, base_url, tunnel=tunnel)
    finally:
        stop(tunnel)
        if reader is not None:
            reader.join(timeout=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lan", "tunnel"), default="lan")
    parser.add_argument("--skip-network-check", action="store_true",
                        help="don't check whether this Wi-Fi lets other devices reach the laptop")
    args = parser.parse_args(argv)
    try:
        if args.mode == "tunnel":
            return serve_tunnel()
        address = lan_ipv4()
        cert_path, key_path = ensure_certificate(address, ROOT / "certs")
        if not args.skip_network_check:
            Thread(target=check_network, args=(address,), daemon=True).start()
        return serve(
            "0.0.0.0",
            LAN_PORT,
            f"https://{address}:{LAN_PORT}",
            ("--ssl-certfile", str(cert_path), "--ssl-keyfile", str(key_path)),
        )
    except (OSError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def handle_termination(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_termination)
    raise SystemExit(main())
