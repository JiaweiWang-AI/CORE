"""Verified HTTPS and atomic, repeatable public-data downloads."""
import hashlib
import os
from pathlib import Path
import ssl
from urllib.error import URLError
from urllib.request import urlopen


def tls_context():
    explicit = os.environ.get('SSL_CERT_FILE')
    if explicit:
        return ssl.create_default_context(cafile=explicit)
    context = ssl.create_default_context()
    # Some python.org macOS installations have an absent default CA bundle.
    # Load trusted OS roots; never disable certificate or hostname verification.
    if not context.cert_store_stats()['x509_ca']:
        for candidate in ('/etc/ssl/cert.pem', '/etc/ssl/certs/ca-certificates.crt'):
            if Path(candidate).is_file():
                context.load_verify_locations(cafile=candidate)
                break
        else:
            try:
                import certifi
                context.load_verify_locations(cafile=certifi.where())
            except ImportError:
                # Windows can populate trust from its OS store during handshake.
                if os.name != 'nt':
                    raise RuntimeError('No trusted CA bundle found. Set SSL_CERT_FILE to your CA bundle or install certifi.')
    return context


def open_url(request, timeout=120):
    try:
        return urlopen(request, timeout=timeout, context=tls_context())
    except URLError as error:
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            raise RuntimeError('HTTPS certificate verification failed. Configure SSL_CERT_FILE with a trusted CA bundle.') from None
        raise


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, path, expected_sha256=None):
    path = Path(path)
    if path.exists():
        if expected_sha256 is not None and file_hash(path) != expected_sha256:
            raise ValueError('Existing file has an unexpected checksum: ' + str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    with open_url(url) as response, temporary.open('wb') as stream:
        for chunk in iter(lambda: response.read(1024*1024), b''):
            stream.write(chunk)
    if expected_sha256 is not None and file_hash(temporary) != expected_sha256:
        raise ValueError('Downloaded file checksum mismatch: ' + path.name)
    temporary.replace(path)
    return path
