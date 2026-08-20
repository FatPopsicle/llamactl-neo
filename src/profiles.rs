use crate::config::{Paths, atomic_json};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Profiles {
    pub expose: Vec<String>,
    pub profiles: BTreeMap<String, Map<String, Value>>,
    pub models: BTreeMap<String, Value>,
}
impl Default for Profiles {
    fn default() -> Self {
        Self {
            expose: vec!["*".into()],
            profiles: BTreeMap::new(),
            models: BTreeMap::new(),
        }
    }
}
impl Profiles {
    pub fn load(paths: &Paths) -> Result<Self> {
        if !paths.profiles.exists() {
            let p = Self::default();
            p.save(paths)?;
            return Ok(p);
        }
        serde_json::from_str(&std::fs::read_to_string(&paths.profiles)?)
            .with_context(|| format!("invalid JSON in {}", paths.profiles.display()))
    }
    pub fn save(&self, paths: &Paths) -> Result<()> {
        atomic_json(&paths.profiles, self)
    }
    pub fn clone_profile(&mut self, source: &str, target: &str) -> Result<()> {
        valid_name(target)?;
        if self.profiles.contains_key(target) {
            bail!("profile '{target}' already exists")
        }
        let value = self
            .profiles
            .get(source)
            .with_context(|| format!("unknown profile '{source}'"))?
            .clone();
        self.profiles.insert(target.into(), value);
        Ok(())
    }
    pub fn rename(&mut self, source: &str, target: &str) -> Result<()> {
        valid_name(target)?;
        if self.profiles.contains_key(target) {
            bail!("profile '{target}' already exists")
        }
        let profile = self
            .profiles
            .remove(source)
            .with_context(|| format!("unknown profile '{source}'"))?;
        self.profiles.insert(target.into(), profile);
        for binding in self.models.values_mut() {
            if binding.as_str() == Some(source) {
                *binding = Value::String(target.into());
            } else if let Some(obj) = binding.as_object_mut()
                && obj.get("profile").and_then(Value::as_str) == Some(source)
            {
                obj.insert("profile".into(), Value::String(target.into()));
            }
        }
        for item in &mut self.expose {
            if item == source {
                *item = target.into();
            }
        }
        Ok(())
    }
    pub fn remove(&mut self, name: &str) -> Result<()> {
        self.profiles
            .remove(name)
            .with_context(|| format!("unknown profile '{name}'"))?;
        self.models.retain(|_, v| {
            v.as_str() != Some(name) && v.get("profile").and_then(Value::as_str) != Some(name)
        });
        Ok(())
    }
    pub fn args(&self, name: &str) -> Result<Vec<String>> {
        self.args_checked(name, None)
    }
    pub fn args_checked(
        &self,
        name: &str,
        known: Option<&BTreeSet<String>>,
    ) -> Result<Vec<String>> {
        const RESERVED: &[&str] = &[
            "m",
            "model",
            "mmproj",
            "model-url",
            "hf",
            "hf-repo",
            "hf-file",
            "host",
            "port",
            "api-key",
            "alias",
            "help",
            "usage",
            "version",
        ];
        let p = self
            .profiles
            .get(name)
            .with_context(|| format!("unknown profile '{name}'"))?;
        let mut args = vec![];

        for key in ["jinja", "no-jinja"] {
            if let Some(value) = p.get(key) {
                append_flag(&mut args, key, value);
            }
        }
        for (key, value) in p {
            if key.starts_with('_') || key == "jinja" || key == "no-jinja" {
                continue;
            }
            let key = key.trim_start_matches('-');
            if RESERVED.contains(&key) {
                bail!("profile '{name}': --{key} is managed by llamactl")
            }
            if !self.is_exposed(key) {
                bail!("profile '{name}': --{key} is not exposed")
            }
            if known.is_some_and(|flags| !flags.contains(key)) {
                continue;
            }
            append_flag(&mut args, key, value);
        }
        if let Some(extra) = p.get("_extra_args").and_then(Value::as_array) {
            let values = extra.iter().filter_map(Value::as_str).collect::<Vec<_>>();
            for token in &values {
                if let Some(flag) = token
                    .strip_prefix("--")
                    .map(|value| value.split('=').next().unwrap_or(value))
                {
                    if RESERVED.contains(&flag) {
                        bail!("profile '{name}': --{flag} is managed by llamactl")
                    }
                    if !self.is_exposed(flag) {
                        bail!("profile '{name}': --{flag} is not exposed")
                    }
                    if known.is_some_and(|flags| !flags.contains(flag)) {
                        bail!("profile '{name}': --{flag} is unknown to installed llama.cpp")
                    }
                }
            }
            args.extend(values.into_iter().map(str::to_owned));
        }
        Ok(args)
    }
    pub fn owner(&self, name: &str) -> Option<&str> {
        self.profiles.get(name)?.get("_model")?.as_str()
    }
    pub fn runtime(&self, name: &str) -> Option<&str> {
        self.profiles
            .get(name)?
            .get("_runtime")?
            .as_str()
            .filter(|runtime| !runtime.is_empty())
    }
    pub fn binding(
        &self,
        model: &str,
        relative: &str,
    ) -> Result<Option<(String, Map<String, Value>)>> {
        for (pattern, binding) in &self.models {
            if !glob_matches(pattern, model) && !glob_matches(pattern, relative) {
                continue;
            }
            let (name, overrides) = if let Some(name) = binding.as_str() {
                (name.to_owned(), Map::new())
            } else if let Some(object) = binding.as_object() {
                let name = object
                    .get("profile")
                    .and_then(Value::as_str)
                    .context("model profile binding lacks 'profile'")?
                    .to_owned();
                let overrides = object
                    .iter()
                    .filter(|(key, _)| key.as_str() != "profile")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                (name, overrides)
            } else {
                bail!("model binding '{pattern}' must be a profile name or object")
            };
            let owner = self
                .owner(&name)
                .with_context(|| format!("profile '{name}' has no owner"))?;
            if owner != model {
                bail!("profile '{name}' belongs to '{owner}', not '{model}'")
            }
            return Ok(Some((name, overrides)));
        }
        Ok(None)
    }
    pub fn environment(&self, name: &str) -> Result<Vec<(String, String)>> {
        let Some(value) = self.profiles.get(name).and_then(|p| p.get("_env")) else {
            return Ok(vec![]);
        };
        let object = match value {
            Value::Object(object) => object.clone(),
            Value::String(text) => {
                if text.trim_start().starts_with('{') {
                    serde_json::from_str::<Map<String, Value>>(text)
                        .context("invalid profile environment JSON")?
                } else {
                    text.lines()
                        .filter(|line| !line.trim().is_empty())
                        .map(|line| {
                            let (key, value) = line
                                .split_once('=')
                                .with_context(|| format!("environment entry lacks '=': {line}"))?;
                            Ok((key.trim().to_owned(), Value::String(value.to_owned())))
                        })
                        .collect::<Result<Map<String, Value>>>()?
                }
            }
            _ => bail!("profile _env must be an object or KEY=VALUE text"),
        };
        object
            .into_iter()
            .map(|(key, value)| {
                let valid = key.chars().enumerate().all(|(index, ch)| {
                    ch == '_' || ch.is_ascii_alphanumeric() && (index > 0 || !ch.is_ascii_digit())
                });
                if !valid || key.is_empty() {
                    bail!("invalid environment variable name '{key}'")
                }
                let value = value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string());
                if value.contains('\0') {
                    bail!("environment variable '{key}' contains NUL")
                }
                Ok((key, value))
            })
            .collect()
    }
    pub fn is_exposed(&self, flag: &str) -> bool {
        self.expose.iter().any(|pattern| {
            pattern == "*"
                || pattern == flag
                || pattern
                    .strip_suffix('*')
                    .is_some_and(|prefix| flag.starts_with(prefix))
        })
    }
}
fn glob_matches(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let parts = pattern.split('*').collect::<Vec<_>>();
    if parts.len() == 1 {
        return pattern == value;
    }
    let mut remainder = value;
    for (index, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if index == 0 && !remainder.starts_with(part) {
            return false;
        }
        let Some(position) = remainder.find(part) else {
            return false;
        };
        remainder = &remainder[position + part.len()..];
    }
    pattern.ends_with('*') || remainder.is_empty()
}

fn append_flag(out: &mut Vec<String>, key: &str, value: &Value) {
    match value {
        Value::Bool(true) => out.push(format!("--{key}")),
        Value::Array(values) => {
            for v in values {
                out.push(format!("--{key}"));
                out.push(display(v));
            }
        }
        Value::Null | Value::Bool(false) => {}
        _ => {
            out.push(format!("--{key}"));
            out.push(display(value));
        }
    }
}
fn display(v: &Value) -> String {
    v.as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| v.to_string())
}
pub fn valid_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name.chars().next().unwrap().is_ascii_alphanumeric()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
    if !valid {
        bail!("profile names use letters, numbers, dots, underscores, or hyphens")
    }
    Ok(())
}
