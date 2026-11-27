import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from cryptography import x509

from scripts import start_https


def test_local_certificate_is_reused_for_same_ip_and_replaced_for_new_ip(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(start_https.shutil, "which", lambda name: None)

    cert_path, key_path = start_https.ensure_certificate("192.168.1.20", tmp_path)
    original = cert_path.read_bytes(), key_path.read_bytes()
    assert start_https.ensure_certificate("192.168.1.20", tmp_path) == (
        cert_path,
        key_path,
    )
    assert (cert_path.read_bytes(), key_path.read_bytes()) == original

    start_https.ensure_certificate("192.168.1.21", tmp_path)
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert ipaddress.ip_address("192.168.1.21") in san.get_values_for_type(
        x509.IPAddress
    )
    assert ipaddress.ip_address("192.168.1.20") not in san.get_values_for_type(
        x509.IPAddress
    )
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    if os.name == "posix":
        assert key_path.stat().st_mode & 0o077 == 0


def test_lan_uses_route_source_ipv4(monkeypatch):
    class RouteProbe:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def connect(self, endpoint):
            assert endpoint[1] > 0

        def getsockname(self):
            return "192.168.1.20", 12345

    monkeypatch.setattr(socket, "socket", lambda family, kind: RouteProbe())
    assert start_https.lan_ipv4() == "192.168.1.20"


def test_certificate_prefers_mkcert_when_available(tmp_path, monkeypatch):
    address = "192.168.1.20"
    monkeypatch.setattr(start_https.shutil, "which", lambda name: "mkcert")

    def run(command, *, check, timeout):
        assert command[0:2] == ["mkcert", "-cert-file"]
        assert command[3] == "-key-file"
        assert command[5:] == [address, "localhost", "127.0.0.1"]
        assert check and timeout > 0
        start_https.write_self_signed_certificate(Path(command[2]), Path(command[4]), address)

    monkeypatch.setattr(start_https.subprocess, "run", run)
    cert_path, key_path = start_https.ensure_certificate(address, tmp_path)
    assert start_https.certificate_is_usable(cert_path, key_path, address)


def test_lan_passes_tls_paths_and_lan_origin_to_uvicorn(tmp_path, monkeypatch, capsys):
    popen = subprocess.Popen
    monkeypatch.setattr(start_https, "ROOT", tmp_path)
    monkeypatch.setattr(start_https, "lan_ipv4", lambda: "192.168.1.20")
    monkeypatch.setattr(start_https, "check_network", lambda address: None)
    monkeypatch.setattr(start_https.shutil, "which", lambda name: None)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://stale.example")

    def launch(command, *, cwd, env):
        assert cwd == tmp_path
        assert env["PUBLIC_BASE_URL"] == "https://192.168.1.20:8443"
        assert command == [
            sys.executable, "-m", "uvicorn", "backend.server.app:app", "--host",
            "0.0.0.0", "--port", "8443", "--ssl-certfile",
            str(tmp_path / "certs" / "lan.crt"), "--ssl-keyfile",
            str(tmp_path / "certs" / "lan.key"),
        ]
        return popen([sys.executable, "-c", "pass"])

    monkeypatch.setattr(start_https.subprocess, "Popen", launch)
    assert start_https.main([]) == 0
    assert "https://192.168.1.20:8443/phone.html" in capsys.readouterr().out


def test_tunnel_waits_for_valid_origin_and_stops_cloudflared(tmp_path, monkeypatch, capsys):
    server_env = tmp_path / "env.json"
    popen = subprocess.Popen
    children = []
    monkeypatch.setattr(start_https.shutil, "which", lambda name: "/fake/cloudflared")

    def launch(command, **kwargs):
        if not children:
            assert command == [
                "/fake/cloudflared", "tunnel", "--url", "http://127.0.0.1:8000",
            ]
            script = """
import sys
import time
print('https://fake.trycloudflare.com.invalid', file=sys.stderr, flush=True)
print('Public URL: https://swarm.trycloudflare.com', file=sys.stderr, flush=True)
time.sleep(30)
"""
        else:
            assert command == [
                sys.executable, "-m", "uvicorn", "backend.server.app:app", "--host",
                "127.0.0.1", "--port", "8000",
            ]
            script = (
                "import json, os; from pathlib import Path; "
                f"Path({str(server_env)!r}).write_text(json.dumps(os.environ['PUBLIC_BASE_URL']))"
            )
        child = popen([sys.executable, "-c", script], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(start_https.subprocess, "Popen", launch)
    assert start_https.main(["--mode", "tunnel"]) == 0
    assert json.loads(server_env.read_text()) == "https://swarm.trycloudflare.com"
    assert "https://swarm.trycloudflare.com/phone.html" in capsys.readouterr().out
    assert all(child.poll() is not None for child in children)


def test_tunnel_times_out_without_launching_uvicorn(monkeypatch, capsys):
    popen = subprocess.Popen
    children = []
    monkeypatch.setattr(start_https.shutil, "which", lambda name: "/fake/cloudflared")
    monkeypatch.setattr(start_https, "TUNNEL_START_TIMEOUT", 0.1)

    def launch(command, **kwargs):
        assert command[0] == "/fake/cloudflared"
        child = popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(start_https.subprocess, "Popen", launch)
    assert start_https.main(["--mode", "tunnel"]) == 1
    assert "timed out" in capsys.readouterr().err.lower()
    assert len(children) == 1
    assert children[0].poll() is not None


def test_tunnel_requires_cloudflared(monkeypatch, capsys):
    monkeypatch.setattr(start_https.shutil, "which", lambda name: None)
    assert start_https.main(["--mode", "tunnel"]) == 1
    assert "Install cloudflared" in capsys.readouterr().err


ARP = """? (172.30.220.1) at 0:11:22:33:44:55 on en0 ifscope [ethernet]
? (172.30.220.6) at c2:59:20:c0:d9:32 on en0 ifscope [ethernet]
? (172.30.220.71) at 2a:51:89:16:9c:91 on en0 ifscope [ethernet]
? (172.30.220.78) at 26:39:f8:d0:7e:a9 on en0 ifscope permanent [ethernet]
? (172.30.220.90) at (incomplete) on en0 ifscope [ethernet]
? (172.30.220.255) at ff:ff:ff:ff:ff:ff on en0 ifscope [ethernet]
? (224.0.0.251) at 1:0:5e:0:0:fb on en0 ifscope permanent [ethernet]
? (10.0.0.4) at aa:bb:cc:dd:ee:ff on en1 ifscope [ethernet]
"""


def test_arp_neighbors_keeps_real_devices_on_the_laptops_subnet():
    assert start_https.arp_neighbors(ARP, "172.30.220.78") == ["172.30.220.1", "172.30.220.6", "172.30.220.71"]


def test_isolation_warning_only_when_no_peer_answers():
    peers = ["172.30.220.1", "172.30.220.6", "172.30.220.71"]
    warning = start_https.isolation_warning(peers, "172.30.220.1", lambda ip: ip == "172.30.220.1")
    assert warning and "hotspot" in warning and "--mode tunnel" in warning
    assert start_https.isolation_warning(peers, "172.30.220.1", lambda ip: ip == "172.30.220.71") is None
    assert start_https.isolation_warning(["172.30.220.1"], "172.30.220.1", lambda ip: False) is None  # router only
    assert start_https.isolation_warning([], None, lambda ip: False) is None
