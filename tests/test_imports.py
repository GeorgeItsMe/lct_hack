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


def test_journal_as_delivered_with_separate_date_and_time():
    content = (
        "ид_события,ид_канала_данных,дата,время,значение_датчика,тревожное\n"
        "10,1,2026-06-15,11:58:00,Норма,f\n"
        "11,1,2026-06-15,11:59:30,Неисправен,t\n"
    ).encode()
    result, cutoff, quality = normalize_events(read_events(content, "csv"), {1})
    assert cutoff == pd.Timestamp("2026-06-15T12:00")
    assert quality["as_of_auto"] and quality["accepted_rows"] == 2
    assert result.alarm.tolist() == [False, True]


def test_appendix_layout_semicolon_cp1251_without_alarm_flag():
    content = (
        "ИД записи журнала;ИД канала данных;ИД типа канала данных;Текущее значение;Дата записи\n"
        "1;1;12;25,40;19.10.2026 12:15\n"
        "2;1;4;Обнаружен дым;19.10.2026 12:16\n"
    ).encode("cp1251")
    result, cutoff, quality = normalize_events(read_events(content, "csv"), {1})
    assert cutoff == pd.Timestamp("2026-10-19T12:20")
    assert result.ts.tolist() == [pd.Timestamp("2026-10-19T12:15"), pd.Timestamp("2026-10-19T12:16")]
    assert quality["alarm_inferred"] and result.alarm.tolist() == [False, True]
    assert result.numeric_value.iloc[0] == 25.4


def test_zip_with_one_journal_is_accepted():
    import zipfile

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("journal.csv", "channel_id,ts,value,alarm\n1,2026-06-15T11:59:00,Норма,false\n")
    result, _, _ = normalize_events(read_events(buffer.getvalue(), "zip"), {1}, "2026-06-15T12:00")
    assert len(result) == 1


def test_a_few_bad_lines_are_skipped_but_a_broken_file_is_refused():
    rows = [sample(ts=f"2026-06-15T11:{m:02d}:00") for m in range(40)] + [sample(ts="сломано")]
    result, _, quality = normalize_events(pd.DataFrame(rows), {1}, "2026-06-15T12:00")
    assert len(result) == 40 and quality["invalid_rows"] == 1
    broken = [sample(ts="сломано")] * 10 + [sample()]
    with pytest.raises(ValueError, match="Невалидных строк: 10 из 11"):
        normalize_events(pd.DataFrame(broken), {1}, "2026-06-15T12:00")


def test_rows_after_an_explicit_moment_are_reported_not_fatal():
    rows = [sample(), sample(ts="2026-06-15T12:30:00")]
    result, _, quality = normalize_events(pd.DataFrame(rows), {1}, "2026-06-15T12:00")
    assert len(result) == 1 and quality["late_rows"] == 1


def test_missing_columns_are_named_in_russian():
    with pytest.raises(ValueError, match="значение"):
        read_events("ид_канала_данных,дата,время\n1,2026-06-15,11:00:00\n".encode(), "csv")


def test_stream_still_refuses_future_rows_whole():
    rows = [sample(), sample(ts="2026-06-15T12:30:00")]
    with pytest.raises(ValueError, match="предшествовать"):
        normalize_events(pd.DataFrame(rows), {1}, "2026-06-15T12:00", strict_time=True)


def test_utf8_letter_cut_at_the_sniffing_boundary_is_not_mistaken_for_cp1251():
    header = "ид_канала_данных,дата,время,значение_датчика,тревожное\n".encode()
    filler = b"1,2026-06-15,11:58:00,0,f\n"
    prefix = b"1,2026-06-15,11:58:00,"
    body = header + filler * ((65535 - len(header) - len(prefix)) // len(filler))
    body += prefix + b"0" * (65535 - len(body) - len(prefix))
    assert len(body) == 65535
    content = body + "Я,f\n".encode() + "1,2026-06-15,11:59:00,Обнаружено движение,f\n".encode() * 10
    # The two bytes of «Я» sit at offsets 65535 and 65536: a naive 64 KiB decode fails as UTF-8.
    with pytest.raises(UnicodeDecodeError):
        content[:65536].decode("utf-8")
    frame = read_events(content, "csv")
    assert list(frame.columns[:2]) == ["channel_id", "date"] and frame.value.iloc[-1] == "Обнаружено движение"
