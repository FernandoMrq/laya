//! Typed questions and the insertion-ordered set they are asked in.
//!
//! Ports `laya/presets.py`'s question shape. Order is load-bearing throughout this module: batch
//! row order follows `QuestionSet` order, and choice label order decides which logit each
//! label's probability comes from (see `sequence-construction.md`).

use crate::error::{LayaError, Result};
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::HashSet;

/// The three question types Laya answers. Values match Python's `QTYPES` and the `qtype` tensor
/// encoding (`model-contract.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionType {
    /// Pick one label from a set of named options.
    Choice = 0,
    /// Place the state on an ordered scale; the answer is an expected level.
    Score = 1,
    /// Does the statement hold? The answer is a probability.
    Noul = 2,
}

// The Python type name (`SequenceBuilder::type_name`) and the `qtype` tensor's integer encoding
// (a plain `qtype as i64` cast, since the discriminants above already match Python's `QTYPES`)
// live in `sequence_builder.rs` alongside the rest of that logic, rather than here, keeping this
// enum bare.

impl TryFrom<i64> for QuestionType {
    type Error = LayaError;

    /// The question type for a `qtype` tensor value. Rust has no implicit enum-from-integer cast,
    /// so this is the explicit equivalent.
    fn try_from(index: i64) -> Result<Self> {
        match index {
            0 => Ok(QuestionType::Choice),
            1 => Ok(QuestionType::Score),
            2 => Ok(QuestionType::Noul),
            other => Err(LayaError::InvalidQuestion(format!(
                "unknown qtype index {other}"
            ))),
        }
    }
}

/// A [`QuestionType::Choice`] question: ordered, named options.
#[derive(Debug, Clone)]
pub struct ChoiceQuestion {
    instructions: Value,
    options: Vec<(String, Value)>,
    option_order: Option<Vec<usize>>,
}

impl ChoiceQuestion {
    /// The prompt: a string verbatim, or anything else rendered through
    /// [`crate::python_json::instructions`].
    pub fn instructions(&self) -> &Value {
        &self.instructions
    }

    /// The options in label order. Index N here is the label for logit N.
    pub fn options(&self) -> &[(String, Value)] {
        &self.options
    }

    /// The labels, in order.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.options.iter().map(|(label, _)| label.as_str())
    }
}

/// A [`QuestionType::Score`] question: an ordered scale, lowest level first.
#[derive(Debug, Clone)]
pub struct ScoreQuestion {
    instructions: Value,
    levels: Vec<Value>,
    option_order: Option<Vec<usize>>,
}

impl ScoreQuestion {
    /// The prompt. See [`ChoiceQuestion::instructions`].
    pub fn instructions(&self) -> &Value {
        &self.instructions
    }

    /// The levels, lowest first. The answer is an expectation over these indices.
    pub fn levels(&self) -> &[Value] {
        &self.levels
    }
}

/// A [`QuestionType::Noul`] question: does the statement hold?
#[derive(Debug, Clone)]
pub struct NoulQuestion {
    instructions: Value,
    if_false: Value,
    if_true: Value,
    labels: Option<(String, String)>,
    option_order: Option<Vec<usize>>,
}

impl NoulQuestion {
    /// The words shown in front of the false and true options: `("false", "true")` unless
    /// [`Question::with_noul_labels`] set others. Only the text the model reads changes; the
    /// answer is still P(true). Ports `_resolve_noul_labels` (`laya/common.py`).
    pub fn labels(&self) -> (&str, &str) {
        match &self.labels {
            Some((false_label, true_label)) => (false_label, true_label),
            None => ("false", "true"),
        }
    }

    /// The prompt. See [`ChoiceQuestion::instructions`].
    pub fn instructions(&self) -> &Value {
        &self.instructions
    }

    /// Wording for the false option, or `Value::Null` for the default.
    pub fn if_false(&self) -> &Value {
        &self.if_false
    }

    /// Wording for the true option, or `Value::Null` for the default.
    pub fn if_true(&self) -> &Value {
        &self.if_true
    }
}

/// One typed question. Construct with [`Question::choice`], [`Question::choice_labels`],
/// [`Question::score`] or [`Question::noul`].
#[derive(Debug, Clone)]
pub enum Question {
    /// See [`ChoiceQuestion`].
    Choice(ChoiceQuestion),
    /// See [`ScoreQuestion`].
    Score(ScoreQuestion),
    /// See [`NoulQuestion`].
    Noul(NoulQuestion),
}

impl Question {
    /// Which of the three question types this is.
    pub fn question_type(&self) -> QuestionType {
        match self {
            Question::Choice(_) => QuestionType::Choice,
            Question::Score(_) => QuestionType::Score,
            Question::Noul(_) => QuestionType::Noul,
        }
    }

    /// The prompt. A string is used verbatim; anything else is serialized with
    /// [`crate::python_json::instructions`], matching Python's `_to_internal`.
    pub fn instructions(&self) -> &Value {
        match self {
            Question::Choice(c) => c.instructions(),
            Question::Score(s) => s.instructions(),
            Question::Noul(n) => n.instructions(),
        }
    }

    /// This question as a choice question.
    pub fn as_choice(&self) -> Option<&ChoiceQuestion> {
        match self {
            Question::Choice(c) => Some(c),
            _ => None,
        }
    }

    /// This question as a score question.
    pub fn as_score(&self) -> Option<&ScoreQuestion> {
        match self {
            Question::Score(s) => Some(s),
            _ => None,
        }
    }

    /// This question as a noul question.
    pub fn as_noul(&self) -> Option<&NoulQuestion> {
        match self {
            Question::Noul(n) => Some(n),
            _ => None,
        }
    }

    /// A choice question over ordered `(label, description)` options. A `Value::Null` description
    /// means "no description"; note that `0` and `false` are real descriptions and are rendered
    /// (see [`crate::python_json::is_blank_criterion`]).
    ///
    /// Rejects duplicate labels and an empty option list: Python would silently collapse
    /// duplicate keys into one option, changing the option count and so every marker position, so
    /// this refuses rather than silently answering a different question.
    pub fn choice<I, L, D>(instructions: impl Into<Value>, options: I) -> Result<Question>
    where
        I: IntoIterator<Item = (L, D)>,
        L: Into<String>,
        D: Into<Value>,
    {
        let mut list = Vec::new();
        let mut seen = HashSet::new();
        for (label, description) in options {
            let label = label.into();
            if !seen.insert(label.clone()) {
                return Err(LayaError::InvalidQuestion(format!(
                    "duplicate choice label '{label}'"
                )));
            }
            list.push((label, description.into()));
        }
        if list.is_empty() {
            return Err(LayaError::InvalidQuestion(
                "a choice question needs at least one option".to_string(),
            ));
        }
        Ok(Question::Choice(ChoiceQuestion {
            instructions: instructions.into(),
            options: list,
            option_order: None,
        }))
    }

    /// A choice question over bare labels, with no descriptions.
    pub fn choice_labels<I, L>(instructions: impl Into<Value>, labels: I) -> Result<Question>
    where
        I: IntoIterator<Item = L>,
        L: Into<String>,
    {
        Question::choice(
            instructions,
            labels.into_iter().map(|label| (label, Value::Null)),
        )
    }

    /// A score question over ordered levels, lowest first. Rejects an empty level list.
    pub fn score<I, D>(instructions: impl Into<Value>, levels: I) -> Result<Question>
    where
        I: IntoIterator<Item = D>,
        D: Into<Value>,
    {
        let list: Vec<Value> = levels.into_iter().map(Into::into).collect();
        if list.is_empty() {
            return Err(LayaError::InvalidQuestion(
                "a score question needs at least one level".to_string(),
            ));
        }
        Ok(Question::Score(ScoreQuestion {
            instructions: instructions.into(),
            levels: list,
            option_order: None,
        }))
    }

    /// A yes/no question. Supplying `if_false` / `if_true` (e.g. `Some("on fire")`, or `None::<Value>`
    /// for the default) replaces the default "no, the statement does not hold" / "yes, the
    /// statement holds" wording.
    pub fn noul(
        instructions: impl Into<Value>,
        if_false: impl Into<Value>,
        if_true: impl Into<Value>,
    ) -> Question {
        Question::Noul(NoulQuestion {
            instructions: instructions.into(),
            if_false: if_false.into(),
            if_true: if_true.into(),
            labels: None,
            option_order: None,
        })
    }

    /// How many options the question renders to: one per choice label or score level, and the
    /// `[false, true]` pair for a noul. Ports `_option_count` (`laya/agent.py`).
    pub fn option_count(&self) -> usize {
        match self {
            Question::Choice(c) => c.options.len(),
            Question::Score(s) => s.levels.len(),
            Question::Noul(_) => 2,
        }
    }

    /// The slot order the options are shown in, when one was set: slot `s` shows option
    /// `order[s]`. `None` is the canonical order.
    pub fn option_order(&self) -> Option<&[usize]> {
        match self {
            Question::Choice(c) => c.option_order.as_deref(),
            Question::Score(s) => s.option_order.as_deref(),
            Question::Noul(n) => n.option_order.as_deref(),
        }
    }

    /// Show the options in another order: slot `s` shows option `order[s]`. Answers still index
    /// by option, so only what the model reads changes. Ports the `option_order` key of a
    /// question definition (`Agent._check_question` and `build_head`, laya 0.4.6): anything but a
    /// permutation of `0..option_count` is refused, since it would drop an option or show one
    /// twice.
    pub fn with_option_order(mut self, order: Vec<usize>) -> Result<Question> {
        let n = self.option_count();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        if sorted != (0..n).collect::<Vec<_>>() {
            return Err(LayaError::InvalidQuestion(format!(
                "'option_order' must be a permutation of range({n}) -- one slot per option, \
                 each option once -- got {order:?}"
            )));
        }
        let slot = match &mut self {
            Question::Choice(c) => &mut c.option_order,
            Question::Score(s) => &mut s.option_order,
            Question::Noul(q) => &mut q.option_order,
        };
        *slot = Some(order);
        Ok(self)
    }

    /// Word a noul's two options with other labels than `false` / `true`, as Python's `labels`
    /// key does. Labels are trimmed and must be distinct and non-empty; only a noul takes them.
    /// Ports `_resolve_noul_labels` (`laya/common.py`).
    pub fn with_noul_labels(
        mut self,
        false_label: impl Into<String>,
        true_label: impl Into<String>,
    ) -> Result<Question> {
        let Question::Noul(noul) = &mut self else {
            return Err(LayaError::InvalidQuestion(
                "'labels' is only supported for noul questions".to_string(),
            ));
        };
        let false_label = false_label.into().trim().to_string();
        let true_label = true_label.into().trim().to_string();
        if false_label.is_empty() || true_label.is_empty() || false_label == true_label {
            return Err(LayaError::InvalidQuestion(
                "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
                    .to_string(),
            ));
        }
        noul.labels = Some((false_label, true_label));
        Ok(self)
    }
}

/// Puts a slot-ordered probability row back into option order: slot `s` showed option
/// `order[s]`. A missing or mismatched order returns the row unchanged. Ports `unpermute_probs`
/// (`laya/common.py`).
pub fn unpermute_probabilities(p: Vec<f64>, order: Option<&[usize]>) -> Vec<f64> {
    match order {
        Some(order) if order.len() == p.len() => {
            let mut canonical = vec![0.0; p.len()];
            for (slot, &option) in order.iter().enumerate() {
                canonical[option] = p[slot];
            }
            canonical
        }
        _ => p,
    }
}

/// An insertion-ordered set of question id to [`Question`], answered in one batch.
///
/// Deliberately backed by [`IndexMap`] rather than a hash map: the batch row order must match the
/// order the caller wrote the questions in, because that is what Python's `dict` guarantees and
/// what the answer order is compared against.
#[derive(Debug, Clone, Default)]
pub struct QuestionSet {
    map: IndexMap<String, Question>,
}

impl QuestionSet {
    /// An empty set.
    pub fn new() -> Self {
        Self {
            map: IndexMap::new(),
        }
    }

    /// How many questions are in the set.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the set has no questions.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Insert a question by id. Returns an error if the id is already present, rather than
    /// silently overwriting it and shifting every subsequent row's meaning.
    pub fn insert(&mut self, id: impl Into<String>, question: Question) -> Result<()> {
        let id = id.into();
        if self.map.contains_key(&id) {
            return Err(LayaError::InvalidQuestion(format!(
                "duplicate question id '{id}'"
            )));
        }
        self.map.insert(id, question);
        Ok(())
    }

    /// Builder-style [`insert`](Self::insert): `QuestionSet::new().with("a", q1)?.with("b", q2)?`.
    pub fn with(mut self, id: impl Into<String>, question: Question) -> Result<Self> {
        self.insert(id, question)?;
        Ok(self)
    }

    /// The question with this id, if any.
    pub fn get(&self, id: &str) -> Option<&Question> {
        self.map.get(id)
    }

    /// Whether the set contains this id.
    pub fn contains_id(&self, id: &str) -> bool {
        self.map.contains_key(id)
    }

    /// The question ids, in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    /// Every `(id, question)` pair, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Question)> {
        self.map.iter().map(|(id, q)| (id.as_str(), q))
    }
}

impl QuestionSet {
    /// An order-sensitive signature of the set: two sets share it only when they have the same
    /// ids, types, instructions and options in the same order. The router groups requests by it
    /// so equal schemas share a forward pass. Ports `_question_schema` (`laya/router.py`), which
    /// is `json.dumps(questions, sort_keys=False)`: option and key order are positional.
    pub(crate) fn schema_key(&self) -> String {
        let rows: Vec<Value> = self
            .map
            .iter()
            .map(|(id, question)| {
                let body = match question {
                    Question::Choice(q) => Value::Array(vec![
                        Value::from("choice"),
                        q.instructions.clone(),
                        Value::Array(
                            q.options
                                .iter()
                                .map(|(label, text)| {
                                    Value::Array(vec![Value::from(label.as_str()), text.clone()])
                                })
                                .collect(),
                        ),
                    ]),
                    Question::Score(q) => Value::Array(vec![
                        Value::from("score"),
                        q.instructions.clone(),
                        Value::Array(q.levels.clone()),
                    ]),
                    Question::Noul(q) => Value::Array(vec![
                        Value::from("noul"),
                        q.instructions.clone(),
                        q.if_false.clone(),
                        q.if_true.clone(),
                        q.labels.as_ref().map_or(Value::Null, |(f, t)| {
                            Value::from(vec![f.as_str(), t.as_str()])
                        }),
                    ]),
                };
                let order = question
                    .option_order()
                    .map_or(Value::Null, |order| Value::from(order.to_vec()));
                Value::Array(vec![Value::from(id.as_str()), body, order])
            })
            .collect();
        Value::Array(rows).to_string()
    }
}

impl std::ops::Index<&str> for QuestionSet {
    type Output = Question;

    /// Panics if no question carries this id; use [`QuestionSet::get`] for a fallible lookup.
    fn index(&self, id: &str) -> &Question {
        self.map
            .get(id)
            .unwrap_or_else(|| panic!("no question with id '{id}'"))
    }
}
