import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from causeway import db
import causeway.scraper as scraper_mod

@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield

def test_log_startup_gap_no_history(capsys):
    scraper_mod._log_startup_gap()
    out = capsys.readouterr().out + capsys.readouterr().err
    # Just verify it doesn't raise

def test_log_startup_gap_with_history(capsys):
    recent = (datetime.now() - timedelta(minutes=10)).isoformat()
    db.log_scrape(recent, "2701", "/img.jpg", "success")
    scraper_mod._log_startup_gap()
    # No assertion on exact text — just verify it completes without error

def test_fetch_with_backoff_returns_payload():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"items": [{"cameras": []}]}

    with patch("causeway.scraper.requests.get", return_value=mock_response):
        result = scraper_mod._fetch_with_backoff()

    assert result == {"items": [{"cameras": []}]}

def test_fetch_with_backoff_returns_none_on_auth_failure():
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("causeway.scraper.requests.get", return_value=mock_response):
        result = scraper_mod._fetch_with_backoff()

    assert result is None

def test_fetch_with_backoff_retries_on_server_error():
    fail_response = MagicMock()
    fail_response.status_code = 500

    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {"items": []}

    with patch("causeway.scraper.requests.get", side_effect=[fail_response, ok_response]):
        with patch("causeway.scraper.time.sleep"):
            result = scraper_mod._fetch_with_backoff()

    assert result == {"items": []}

def test_run_once_saves_image_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(scraper_mod, "IMAGE_BASE_DIR", str(tmp_path / "traffic_images"))

    payload = {
        "items": [{
            "cameras": [
                {"camera_id": "2701", "image": "http://fake-cdn.test/2701.jpg"}
            ]
        }]
    }

    fake_image_content = b"\xff\xd8\xff"  # minimal JPEG header bytes

    with patch.object(scraper_mod, "_fetch_with_backoff", return_value=payload):
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = fake_image_content
        with patch("causeway.scraper.requests.get", return_value=img_resp):
            scraper_mod.run_once()

    logs = db.get_recent_scrape_logs(limit=10)
    assert len(logs) == 1
    assert logs[0][1] == "2701"
    assert logs[0][3] == "success"
