import json
from io import BytesIO

import pandas as pd
import pytest

from moscollector.features import group_episodes
from moscollector.importing import ImportManager, normalize_events, read_events


def sample(**changes):
    return {"channel_id": 1, "ts": "2026-06-15T11:59:00", "value": "Норма", "alarm": False, **changes}


@pytest.mark.parametrize("extension", ["csv", "xlsx", "json", "xml"])
def test_formats_share_the_same_contract(extension):
    row = sample()
    if extension == "json":
        content = json.dumps([row]).encode()
    elif extension == "xml":
        content = b"<events><event><channel_id>1</channel_id><ts>2026-06-15T11:59:00</ts><value>OK</value><alarm>false</alarm></event></events>"
    elif extension == "csv":
        content = pd.DataFrame([row]).to_csv(index=False).encode()
    else:
        buffer = BytesIO()
        pd.DataFrame([row]).to_excel(buffer, index=False)
        content = buffer.getvalue()
    result, cutoff, quality = normalize_events(read_events(content, extension), {1}, "2026-06-15T12:00")
    assert len(result) == 1 and quality["accepted_rows"] == 1
    assert result.ts.iloc[0] < cutoff
    assert not result.alarm.iloc[0]


@pytest.mark.parametrize(
    "change",
    [
        {"channel_id": 2},
        {"channel_id": 1.5},
        {"ts": "2026-06-15T12:00"},
        {"ts": "nonsense"},
        {"alarm": "maybe"},
        {"value": ""},
    ],
)
def test_invalid_rows_never_enter_inference(change):
    with pytest.raises(ValueError):
        normalize_events(pd.DataFrame([sample(**change)]), {1}, "2026-06-15T12:00")


def test_unknown_channels_are_skipped_and_reported_not_fatal():
    rows = [sample(), sample(channel_id=77), sample(channel_id=77, value="Неисправен"), sample(channel_id=78)]
    result, _, quality = normalize_events(pd.DataFrame(rows), {1}, "2026-06-15T12:00")
    assert result.channel_id.tolist() == [1]
    assert quality["input_rows"] == 4 and quality["accepted_rows"] == 1
    assert quality["unknown_channel_rows"] == 3
    assert quality["unknown_channels"] == [77, 78] and quality["unknown_channel_count"] == 2


def test_timezone_and_duplicate_normalization():
    row = sample(ts="2026-06-15T08:59:00Z")
    result, _, quality = normalize_events(pd.DataFrame([row, row]), {1}, "2026-06-15T12:00")
    assert len(result) == 1
    assert result.ts.iloc[0] == pd.Timestamp("2026-06-15T11:59:00")
    assert quality["exact_duplicates"] == 1


def test_xml_external_entities_are_rejected():
    xml = b'<!DOCTYPE events [<!ENTITY ex SYSTEM "file:///etc/passwd">]><events><event><value>&ex;</value></event></events>'
    with pytest.raises(ValueError):
        read_events(xml, "xml")


def test_empty_episode_set_is_a_valid_healthy_history():
    grouped = group_episodes(pd.DataFrame(columns=["object_id", "kind"]))
    assert grouped.empty and "episode_id" in grouped
    assert pd.api.types.is_datetime64_any_dtype(grouped.start_ts)


def test_many_previews_do_not_hide_the_latest_stream_job(tmp_path):
    for i in range(105):
        directory = tmp_path / str(i)
        directory.mkdir()
        (directory / "status.json").write_text(
            json.dumps(
                {
                    "id": str(i),
                    "created_at": f"{i:04d}",
                    "mode": "accumulated_stream" if i == 0 else "import_preview",
                }
            )
        )
    manager = ImportManager.__new__(ImportManager)
    manager.root = tmp_path
    assert len(manager.list()) == 100
    assert manager.list(mode="accumulated_stream")[0]["id"] == "0"
