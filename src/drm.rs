//! GPU memory reporting via the DRM fdinfo interface.
//!
//! This is the only *cross-vendor* memory interface the kernel defines —
//! `Documentation/gpu/drm-usage-stats.rst`. amdgpu additionally publishes
//! `mem_info_vram_used` in sysfs, which is why AMD boxes can just cat a file,
//! but i915 and xe publish no such thing and NVIDIA publishes neither. On an
//! Intel Arc machine there is no card-wide VRAM counter anywhere: sysfs has
//! none, the xe perf PMU exposes only engine ticks and clocks, and debugfs is
//! unavailable inside an unprivileged container. Summing fdinfo is not a
//! workaround there, it is the only option.
//!
//! The spec is deliberately *per client*: there is no card-wide line. The card
//! total is the sum over clients, deduplicated by `(drm-pdev, drm-client-id)`
//! because a single client normally holds the same buffers open on many fds —
//! counting fds instead of clients inflates the total several-fold.
//!
//! Per-process attribution is the part vendor tools do not give you, and it is
//! what makes a fit calculation honest: a card can be full of *another
//! process's* model, and "free VRAM" alone cannot tell you whether that is
//! something you may evict or something you must not touch.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

/// One DRM client's device-local memory footprint.
///
/// The per-client fields are not read by the telemetry card, which only needs the
/// card-wide sum — but per-process attribution is the capability no vendor tool
/// offers, and it is what a fit calculation needs to distinguish "the card is
/// full" from "the card is full of another process's model". Kept as public API
/// for that consumer rather than collected and thrown away.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DrmClient {
    pub pid: i32,
    pub comm: String,
    /// PCI address, e.g. `0000:03:00.0`
    pub pdev: String,
    /// `drm-total-*`: what this client has allocated in device-local memory.
    /// This is the field nvtop reports, and it is the conservative one — an
    /// allocation that is currently evicted can be paged straight back in, so a
    /// fit calculation that ignores it will happily overcommit the card.
    pub allocated_bytes: u64,
    /// `drm-resident-*`: the subset physically resident right now. Equal to
    /// `allocated_bytes` on an idle machine, lower under memory pressure.
    pub resident_bytes: u64,
}

#[derive(Debug, Default, Clone)]
pub struct DrmMemory {
    pub clients: Vec<DrmClient>,
}

#[allow(dead_code)]
impl DrmMemory {
    /// Card-wide allocated bytes across every client on every device. Prefer
    /// this for capacity decisions — see [`DrmClient::allocated_bytes`].
    pub fn total_allocated(&self) -> u64 {
        self.clients.iter().map(|c| c.allocated_bytes).sum()
    }

    /// Card-wide resident bytes: what is physically on the card now. Not the
    /// number to use when deciding whether another model fits.
    pub fn total_resident(&self) -> u64 {
        self.clients.iter().map(|c| c.resident_bytes).sum()
    }

    /// Resident bytes per PCI device.
    pub fn per_device(&self) -> BTreeMap<String, u64> {
        let mut out = BTreeMap::new();
        for c in &self.clients {
            *out.entry(c.pdev.clone()).or_insert(0) += c.allocated_bytes;
        }
        out
    }

    /// Resident bytes for one process, across all devices.
    pub fn for_pid(&self, pid: i32) -> u64 {
        self.clients
            .iter()
            .filter(|c| c.pid == pid)
            .map(|c| c.allocated_bytes)
            .sum()
    }
}

/// Parse a fdinfo size value: `"5546064 KiB"`, `"3 MiB"`, or a bare `"0"`.
fn parse_size(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    let scale = match parts.next() {
        None => 1,
        Some("B") => 1,
        Some("KiB") => 1024,
        Some("MiB") => 1024 * 1024,
        Some("GiB") => 1024 * 1024 * 1024,
        // Unknown suffix: refuse rather than silently reporting bytes and
        // under-reporting by three orders of magnitude.
        Some(_) => return None,
    };
    Some(n * scale)
}

/// Sum the device-local regions of one fdinfo file.
///
/// Region naming is driver-specific — xe and amdgpu use `vram0`, i915 uses
/// `local0`. Both are matched. `system` and `gtt` are deliberately excluded:
/// they are host memory, and counting them as VRAM is how a tool ends up
/// claiming a 16 GiB card is holding 40 GiB.
fn device_local_bytes(fields: &BTreeMap<String, String>, prefix: &str) -> u64 {
    fields
        .iter()
        .filter(|(k, _)| {
            k.strip_prefix(prefix)
                .is_some_and(|region| region.starts_with("vram") || region.starts_with("local"))
        })
        .filter_map(|(_, v)| parse_size(v))
        .sum()
}

fn read_fdinfo(path: &Path) -> Option<BTreeMap<String, String>> {
    let text = fs::read_to_string(path).ok()?;
    let mut map = BTreeMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    Some(map)
}

/// Walk every process's DRM fds and collect device-local memory per client.
///
/// Processes we cannot read are skipped silently: without root this sees only
/// our own clients, which yields a *lower bound* on card usage rather than an
/// error. Callers that need the true card total must run privileged.
pub fn read() -> DrmMemory {
    let mut out = DrmMemory::default();
    let mut seen: HashSet<(String, String)> = HashSet::new();

    let Ok(procs) = fs::read_dir("/proc") else {
        return out;
    };

    for entry in procs.flatten() {
        let name = entry.file_name();
        let Some(pid_str) = name.to_str() else {
            continue;
        };
        let Ok(pid) = pid_str.parse::<i32>() else {
            continue;
        };

        let Ok(fds) = fs::read_dir(entry.path().join("fdinfo")) else {
            continue;
        };

        let comm = fs::read_to_string(entry.path().join("comm"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        for fd in fds.flatten() {
            let Some(fields) = read_fdinfo(&fd.path()) else {
                continue;
            };
            let (Some(pdev), Some(cid)) = (fields.get("drm-pdev"), fields.get("drm-client-id"))
            else {
                continue;
            };
            // One client, many fds — count each client once.
            if !seen.insert((pdev.clone(), cid.clone())) {
                continue;
            }
            let allocated = device_local_bytes(&fields, "drm-total-");
            let resident = device_local_bytes(&fields, "drm-resident-");
            if allocated == 0 && resident == 0 {
                continue;
            }
            out.clients.push(DrmClient {
                pid,
                comm: comm.clone(),
                pdev: pdev.clone(),
                allocated_bytes: allocated,
                resident_bytes: resident,
            });
        }
    }
    out
}

/// Card-wide VRAM capacity, where the driver bothers to publish it.
///
/// amdgpu exposes `mem_info_vram_total` in sysfs. xe and i915 do not — but xe
/// *does* publish capacity via `DRM_IOCTL_XE_DEVICE_QUERY` on the render node
/// (`DRM_XE_MEM_REGION_CLASS_VRAM` -> `mr.total_size`), which is how nvtop gets
/// it. That ioctl is not implemented here yet, so this returns 0 on Intel and
/// the caller must source capacity elsewhere for now.
pub fn device_total_bytes() -> u64 {
    let Ok(cards) = fs::read_dir("/sys/class/drm") else {
        return 0;
    };
    let mut total = 0;
    for card in cards.flatten() {
        let name = card.file_name();
        let Some(name) = name.to_str() else { continue };
        // `card1` yes, `card1-DP-1` no — connectors are not devices.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let path = card.path().join("device/mem_info_vram_total");
        if let Ok(text) = fs::read_to_string(path) {
            total += text.trim().parse::<u64>().unwrap_or(0);
        }
    }
    total
}

/// GPU temperatures from the DRM devices' hwmon nodes.
///
/// Index is not fixed across drivers: xe starts at `temp2_input`, amdgpu at
/// `temp1_input`, so probe a range rather than assuming.
pub fn temperatures() -> Vec<f64> {
    let mut out = Vec::new();
    let Ok(cards) = fs::read_dir("/sys/class/drm") else {
        return out;
    };
    for card in cards.flatten() {
        let name = card.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let Ok(hwmons) = fs::read_dir(card.path().join("device/hwmon")) else {
            continue;
        };
        for hwmon in hwmons.flatten() {
            for idx in 1..=4 {
                let path = hwmon.path().join(format!("temp{idx}_input"));
                if let Ok(text) = fs::read_to_string(&path)
                    && let Ok(millidegrees) = text.trim().parse::<f64>()
                    && millidegrees > 0.0
                {
                    out.push(millidegrees / 1000.0);
                    break; // one reading per device is enough for a dashboard
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fdinfo_sizes() {
        assert_eq!(parse_size("0"), Some(0));
        assert_eq!(parse_size("3 MiB"), Some(3 * 1024 * 1024));
        assert_eq!(parse_size("5546064 KiB"), Some(5546064 * 1024));
        assert_eq!(parse_size("nonsense"), None);
        // An unrecognised suffix must not be treated as bytes.
        assert_eq!(parse_size("12 QiB"), None);
    }

    #[test]
    fn counts_only_device_local_regions() {
        let mut f = BTreeMap::new();
        f.insert("drm-resident-system".into(), "0".into());
        f.insert("drm-resident-gtt".into(), "3 MiB".into());
        f.insert("drm-resident-vram0".into(), "5546064 KiB".into());
        // gtt and system are host memory and must not count as VRAM.
        assert_eq!(device_local_bytes(&f, "drm-resident-"), 5546064 * 1024);
    }

    #[test]
    fn total_and_resident_are_read_independently() {
        let mut f = BTreeMap::new();
        f.insert("drm-total-vram0".into(), "8 GiB".into());
        f.insert("drm-resident-vram0".into(), "5 GiB".into());
        // Under eviction these diverge; nvtop reports the former.
        assert_eq!(device_local_bytes(&f, "drm-total-"), 8 * (1 << 30));
        assert_eq!(device_local_bytes(&f, "drm-resident-"), 5 * (1 << 30));
    }

    #[test]
    fn i915_local_region_is_recognised() {
        let mut f = BTreeMap::new();
        f.insert("drm-resident-local0".into(), "1 GiB".into());
        assert_eq!(device_local_bytes(&f, "drm-resident-"), 1024 * 1024 * 1024);
    }
}

#[cfg(test)]
mod live {
    use super::*;

    /// Manual smoke test — reads the real machine, so it is #[ignore]d.
    /// `cargo test --release -- --ignored --nocapture drm::live`
    #[test]
    #[ignore]
    fn dump_live_readings() {
        let mem = read();
        println!("per device:");
        for (dev, bytes) in mem.per_device() {
            println!("  {dev}  {:8.2} GiB", bytes as f64 / (1 << 30) as f64);
        }
        println!("per client:");
        for c in &mem.clients {
            println!(
                "  pid {:>6} {:<16} {}  alloc {:7.2} GiB  resident {:7.2} GiB",
                c.pid,
                c.comm,
                c.pdev,
                c.allocated_bytes as f64 / (1 << 30) as f64,
                c.resident_bytes as f64 / (1 << 30) as f64
            );
        }
        println!(
            "card-wide allocated: {:.2} GiB   resident: {:.2} GiB",
            mem.total_allocated() as f64 / (1 << 30) as f64,
            mem.total_resident() as f64 / (1 << 30) as f64
        );
        println!(
            "capacity from sysfs: {} bytes (0 = driver publishes none)",
            device_total_bytes()
        );
        println!("temps: {:?}", temperatures());
    }
}
