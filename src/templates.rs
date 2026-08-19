use crate::config::Paths;
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;


#[derive(Debug, Clone)]
#[derive(Default)]
pub struct Templates {
    pub templates: BTreeMap<String, String>,
}


impl Templates {
    pub fn load(paths: &Paths) -> Result<Self> {
        fs::create_dir_all(&paths.templates)
            .with_context(|| format!("create {}", paths.templates.display()))?;
        let mut templates = BTreeMap::new();
        for entry in fs::read_dir(&paths.templates)
            .with_context(|| format!("read {}", paths.templates.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("jinja") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let content = fs::read_to_string(&path).unwrap_or_default();
            templates.insert(name.to_owned(), content);
        }
        migrate_legacy_json(paths, &mut templates)?;
        Ok(Self { templates })
    }

    pub fn save(&self, paths: &Paths) -> Result<()> {
        fs::create_dir_all(&paths.templates)
            .with_context(|| format!("create {}", paths.templates.display()))?;
        let known = self
            .templates
            .keys()
            .map(|name| format!("{name}.jinja"))
            .collect::<BTreeSet<_>>();
        for entry in fs::read_dir(&paths.templates)
            .with_context(|| format!("read {}", paths.templates.display()))?
        {
            let entry = entry?;
            let file_name = entry.file_name().to_string_lossy().into_owned();
            if file_name.ends_with(".jinja") && !known.contains(&file_name) {
                let _ = fs::remove_file(entry.path());
            }
        }
        for (name, content) in &self.templates {
            fs::write(template_path(paths, name), content)
                .with_context(|| format!("write template {name}"))?;
        }
        Ok(())
    }
}

fn template_path(paths: &Paths, name: &str) -> PathBuf {
    paths.templates.join(format!("{name}.jinja"))
}


fn migrate_legacy_json(paths: &Paths, templates: &mut BTreeMap<String, String>) -> Result<()> {
    let legacy = paths.templates.with_file_name("templates.json");
    if !legacy.is_file() {
        return Ok(());
    }
    let text = fs::read_to_string(&legacy).unwrap_or_default();
    if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(map) = data.get("templates").and_then(serde_json::Value::as_object)
    {
        for (name, value) in map {
            if let Some(content) = value.as_str()
                && !templates.contains_key(name) {
                    let _ = fs::write(template_path(paths, name), content);
                    templates.insert(name.clone(), content.to_owned());
                }
        }
    }
    let _ = fs::rename(&legacy, legacy.with_extension("json.migrated"));
    Ok(())
}

pub fn valid_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name.chars().next().unwrap().is_ascii_alphanumeric()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || " ._-".contains(c));
    if !valid {
        bail!("template names use letters, numbers, spaces, dots, underscores, or hyphens")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_template_names() {
        assert!(valid_name("chatml").is_ok());
        assert!(valid_name("Llama 3 Instruct").is_ok());
        assert!(valid_name("mistral-v0.2").is_ok());
        assert!(valid_name("").is_err());
        assert!(valid_name("  leading").is_err());
        assert!(valid_name("bad/name").is_err());
    }
}
