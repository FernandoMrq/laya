//! Post-hoc calibration that replaces a checkpoint's own: `temperatures.json` beside an export,
//! or [`crate::LayaOptions::temperatures`]. Ports what the Laya server does to its english model
//! (`_apply_temps` in the reference `server.py`): `agent.temperature = [choice, score, noul]` and
//! `agent.temperature_by_options = {}`, so stale per-option buckets stop overriding the refit
//! per-type values.
//!
//! The file is the one the export tool writes:
//! `{"temperature": [choice, score, noul], "replaces_config_calibration": true, ...}`. It applies
//! only when `replaces_config_calibration` is `true`; a file that records the checkpoint's native
//! calibration (`false`) changes nothing.

use crate::error::{LayaError, Result};
use serde_json::Value;
use std::path::Path;

/// The file name looked up in an artifact directory.
pub const FILE_NAME: &str = "temperatures.json";

/// The range the server accepts (`TEMPERATURE_RANGE`); anything outside it is refused rather
/// than clamped, so a bad value never becomes a silently different calibration.
pub const RANGE: (f64, f64) = (0.5, 5.0);

const KEYS: [&str; 3] = ["choice", "score", "noul"];

/// Checks one `[choice, score, noul]` triple against [`RANGE`].
pub fn check(temperatures: [f64; 3]) -> Result<[f64; 3]> {
    let (low, high) = RANGE;
    if temperatures
        .iter()
        .all(|t| t.is_finite() && (low..=high).contains(t))
    {
        Ok(temperatures)
    } else {
        Err(LayaError::InvalidConfig(format!(
            "temperatures must be three numbers in [{low}, {high}] (choice, score, noul), got \
             {temperatures:?}"
        )))
    }
}

/// The replacement a `temperatures.json` text asks for, or `None` when it keeps the
/// checkpoint's calibration. `temperature` is a `[choice, score, noul]` list or an object with
/// those keys, as the server's `_parse_temperature_list` takes it.
pub fn parse(text: &str) -> Result<Option<[f64; 3]>> {
    let invalid = |why: &str| LayaError::InvalidConfig(format!("{FILE_NAME}: {why}"));
    let root: Value = serde_json::from_str(text).map_err(|e| invalid(&e.to_string()))?;
    if root
        .get("replaces_config_calibration")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Ok(None);
    }
    let raw: Vec<Option<f64>> = match root.get("temperature") {
        Some(Value::Array(list)) => list.iter().map(Value::as_f64).collect(),
        Some(Value::Object(map)) => KEYS
            .iter()
            .map(|k| map.get(*k).and_then(Value::as_f64))
            .collect(),
        _ => return Err(invalid("no 'temperature' list")),
    };
    let values: Vec<f64> = raw
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| invalid("'temperature' holds a non-number"))?;
    let triple: [f64; 3] = values
        .try_into()
        .map_err(|_| invalid("'temperature' needs exactly three values"))?;
    check(triple).map(Some)
}

/// [`parse`] of `dir/temperatures.json`; `None` when the file does not exist.
pub fn read(dir: &Path) -> Result<Option<[f64; 3]>> {
    let path = dir.join(FILE_NAME);
    match std::fs::read_to_string(&path) {
        Ok(text) => parse(&text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_replacing_file_gives_its_three_temperatures() {
        let list = r#"{"temperature": [2.23, 1.69, 5.0], "keys": ["choice", "score", "noul"],
                       "source": "default", "replaces_config_calibration": true}"#;
        let object = r#"{"temperature": {"noul": 5.0, "choice": 2.23, "score": 1.69},
                         "replaces_config_calibration": true}"#;

        assert_eq!(parse(list).unwrap(), Some([2.23, 1.69, 5.0]));
        assert_eq!(parse(object).unwrap(), Some([2.23, 1.69, 5.0]));
    }

    #[test]
    fn a_native_file_changes_nothing() {
        let native = r#"{"temperature": [1.0, 1.0, 1.0], "replaces_config_calibration": false}"#;

        assert_eq!(parse(native).unwrap(), None);
        assert_eq!(parse(r#"{"temperature": [9, 9, 9]}"#).unwrap(), None);
    }

    #[test]
    fn a_bad_replacement_is_refused() {
        for text in [
            r#"{"temperature": [2.23, 1.69], "replaces_config_calibration": true}"#,
            r#"{"temperature": [2.23, 1.69, 5.1], "replaces_config_calibration": true}"#,
            r#"{"temperature": [0.1, 1.0, 1.0], "replaces_config_calibration": true}"#,
            r#"{"temperature": ["2", 1.0, 1.0], "replaces_config_calibration": true}"#,
            r#"{"replaces_config_calibration": true}"#,
            "not json",
        ] {
            assert!(
                matches!(parse(text), Err(LayaError::InvalidConfig(_))),
                "{text}"
            );
        }
    }

    #[test]
    fn a_missing_file_is_no_replacement() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read(dir.path()).unwrap(), None);
        std::fs::write(
            dir.path().join(FILE_NAME),
            r#"{"temperature": [3, 2, 1], "replaces_config_calibration": true}"#,
        )
        .unwrap();
        assert_eq!(read(dir.path()).unwrap(), Some([3.0, 2.0, 1.0]));
    }
}
