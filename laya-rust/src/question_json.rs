//! Question definitions in the JSON shape Python takes them (`system_one`, the HTTP `/predict`
//! body): `{id: {"type": ..., "instructions": ..., "criteria": ...}}`. Ports the checks of
//! `Agent._check_question` and the normalisation of `Agent._to_internal` (`laya/agent.py`, laya
//! 0.4.6), so a server can take a request body as it arrives and refuse what Python refuses.

use crate::error::{LayaError, Result};
use crate::laya_router::python_repr_str;
use crate::questions::{Question, QuestionSet};
use serde_json::{Map, Value};

impl QuestionSet {
    /// Builds a question set from a JSON object of question id to definition, in the object's
    /// key order (the batch row order).
    ///
    /// * `choice`: `criteria` is an object of label to description, or a list of labels (each
    ///   with no description). A list label must be a string or an integer; Python also accepts
    ///   floats and booleans there, whose answer keys do not round-trip through JSON, so this
    ///   port refuses them instead of answering under a different key.
    /// * `score`: `criteria` is a non-empty list of levels, none of them null.
    /// * `noul`: `criteria` is omitted or an object keyed only `true` / `false` (any case).
    ///
    /// The keys `labels` and `option_order` are not supported yet and are refused.
    pub fn from_json(questions: &Value) -> Result<QuestionSet> {
        let Value::Object(map) = questions else {
            return Err(LayaError::InvalidQuestion(format!(
                "questions must be a dict of question id -> definition, got {}",
                json_type(questions)
            )));
        };
        let mut set = QuestionSet::new();
        for (id, definition) in map {
            set.insert(id.clone(), question_from_json(id, definition)?)?;
        }
        Ok(set)
    }
}

/// One question definition. See [`QuestionSet::from_json`].
pub fn question_from_json(id: &str, definition: &Value) -> Result<Question> {
    let fail = |message: String| {
        LayaError::InvalidQuestion(format!("question {}: {message}", python_repr_str(id)))
    };
    if id.trim().is_empty() {
        return Err(LayaError::InvalidQuestion(format!(
            "question id must be a non-empty string, got {}",
            python_repr_str(id)
        )));
    }
    let Value::Object(definition) = definition else {
        return Err(fail(format!(
            "definition must be a dict, got {}",
            json_type(definition)
        )));
    };
    let qtype = match definition.get("type") {
        Some(Value::String(t)) if matches!(t.as_str(), "choice" | "score" | "noul") => t.as_str(),
        other => {
            return Err(fail(format!(
                "unknown type {}; use one of ['choice', 'noul', 'score']",
                other.map_or_else(|| "None".to_string(), Value::to_string)
            )));
        }
    };
    let instructions = checked_instructions(definition).map_err(fail)?;
    let criteria = definition.get("criteria").filter(|c| !c.is_null());
    let question = match qtype {
        "choice" => choice_from_json(instructions, criteria).map_err(fail)?,
        "score" => score_from_json(instructions, criteria).map_err(fail)?,
        _ => noul_from_json(instructions, criteria).map_err(fail)?,
    };
    for key in ["option_order", "labels"] {
        if definition.contains_key(key) {
            return Err(fail(format!("'{key}' is not supported by this port yet")));
        }
    }
    Ok(question)
}

fn checked_instructions(definition: &Map<String, Value>) -> std::result::Result<Value, String> {
    const ADD: &str = "add the text the model should answer";
    match definition.get("instructions") {
        None => Err(format!("no 'instructions'; {ADD}")),
        Some(Value::Null) => Err(format!("'instructions' must not be None; {ADD}")),
        Some(Value::String(s)) if s.trim().is_empty() => {
            Err(format!("'instructions' must not be empty; {ADD}"))
        }
        Some(Value::Array(a)) if a.is_empty() => {
            Err(format!("'instructions' must not be empty; {ADD}"))
        }
        Some(Value::Object(o)) if o.is_empty() => {
            Err(format!("'instructions' must not be empty; {ADD}"))
        }
        Some(value) => Ok(value.clone()),
    }
}

fn choice_from_json(
    instructions: Value,
    criteria: Option<&Value>,
) -> std::result::Result<Question, String> {
    let options: Vec<(String, Value)> = match criteria {
        Some(Value::Object(map)) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        Some(Value::Array(labels)) => labels
            .iter()
            .enumerate()
            .map(|(i, label)| match label {
                Value::String(s) => Ok((s.clone(), Value::Null)),
                Value::Number(n) if n.is_i64() || n.is_u64() => Ok((n.to_string(), Value::Null)),
                Value::Null => Err(format!(
                    "choice label {i} is null; a label is rendered as option text and used as \
                     the answer key, so it must be a string, number or bool"
                )),
                other => Err(format!(
                    "choice label {i} is a {}; this port takes string or integer labels, whose \
                     answer keys round-trip through JSON",
                    json_type(other)
                )),
            })
            .collect::<std::result::Result<_, _>>()?,
        _ => {
            return Err(
                "a choice question takes 'criteria' as a dict of label -> description, \
                        or a list of labels"
                    .to_string(),
            );
        }
    };
    if options.is_empty() {
        return Err("a choice question needs at least one criterion".to_string());
    }
    Question::choice(instructions, options).map_err(|e| e.to_string())
}

fn score_from_json(
    instructions: Value,
    criteria: Option<&Value>,
) -> std::result::Result<Question, String> {
    let Some(Value::Array(levels)) = criteria else {
        return Err(
            "a score question takes 'criteria' as a list of level descriptions, index 0 first"
                .to_string(),
        );
    };
    if levels.is_empty() {
        return Err("a score question needs at least one level".to_string());
    }
    if let Some(i) = levels.iter().position(Value::is_null) {
        return Err(format!(
            "score level {i} is null; give every level a description, index 0 first"
        ));
    }
    Question::score(instructions, levels.iter().cloned()).map_err(|e| e.to_string())
}

fn noul_from_json(
    instructions: Value,
    criteria: Option<&Value>,
) -> std::result::Result<Question, String> {
    let (mut if_false, mut if_true) = (Value::Null, Value::Null);
    match criteria {
        None => {}
        Some(Value::Object(map)) => {
            for (key, value) in map {
                // `_to_internal` lowercases the keys, so `True` describes the true option.
                match key.to_lowercase().as_str() {
                    "false" => if_false = value.clone(),
                    "true" => if_true = value.clone(),
                    _ => {
                        let mut keys: Vec<String> = map.keys().map(|k| k.to_lowercase()).collect();
                        keys.sort();
                        keys.dedup();
                        return Err(format!(
                            "a noul question takes 'criteria' keyed only 'true'/'false' (either \
                             or both, and omitted is fine), got {keys:?}"
                        ));
                    }
                }
            }
        }
        Some(_) => {
            return Err("a noul question takes 'criteria' as a dict with optional \
                        'true'/'false' descriptions, or omits it"
                .to_string());
        }
    }
    Ok(Question::noul(instructions, if_false, if_true))
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::questions::QuestionType;
    use serde_json::json;

    #[test]
    fn the_three_types_parse_in_key_order() {
        let set = QuestionSet::from_json(&json!({
            "department": {"type": "choice", "instructions": "Which team?",
                           "criteria": {"billing": "money", "technical": null}},
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["low", "medium", "high"]},
            "churn": {"type": "noul", "instructions": "Will they leave?",
                      "criteria": {"True": "they cancel"}},
            "tags": {"type": "choice", "instructions": "Tag", "criteria": ["a", 7]}
        }))
        .unwrap();

        let ids: Vec<&str> = set.ids().collect();
        assert_eq!(ids, ["department", "urgency", "churn", "tags"]);
        let choice = set["department"].as_choice().unwrap();
        assert_eq!(choice.options()[1], ("technical".to_string(), Value::Null));
        assert_eq!(set["urgency"].question_type(), QuestionType::Score);
        let noul = set["churn"].as_noul().unwrap();
        assert_eq!(noul.if_true(), &json!("they cancel"));
        assert_eq!(noul.if_false(), &Value::Null);
        let labels: Vec<&str> = set["tags"].as_choice().unwrap().labels().collect();
        assert_eq!(labels, ["a", "7"]);
    }

    #[test]
    fn malformed_definitions_are_refused_naming_the_question() {
        let cases = [
            (
                json!({"type": "choice", "instructions": "x", "criteria": {}}),
                "at least one",
            ),
            (
                json!({"type": "choice", "instructions": "x", "criteria": "a"}),
                "takes 'criteria'",
            ),
            (
                json!({"type": "choice", "instructions": "x", "criteria": [null]}),
                "is null",
            ),
            (
                json!({"type": "choice", "instructions": "x", "criteria": [1.5]}),
                "integer labels",
            ),
            (
                json!({"type": "choice", "instructions": "x", "criteria": ["a", "a"]}),
                "duplicate",
            ),
            (
                json!({"type": "score", "instructions": "x", "criteria": {}}),
                "list of level",
            ),
            (
                json!({"type": "score", "instructions": "x", "criteria": ["a", null]}),
                "level 1",
            ),
            (
                json!({"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}),
                "keyed only",
            ),
            (
                json!({"type": "noul", "instructions": "  "}),
                "must not be empty",
            ),
            (json!({"type": "noul"}), "no 'instructions'"),
            (
                json!({"type": "yesno", "instructions": "x"}),
                "unknown type",
            ),
            (json!("noul"), "must be a dict"),
        ];
        for (definition, expected) in cases {
            let error = question_from_json("q", &definition)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{definition}: {error}");
            assert!(
                error.starts_with("question 'q'") || error.contains("duplicate"),
                "{error}"
            );
        }
        assert!(QuestionSet::from_json(&json!([])).is_err());
    }
}
