//! Minimal, lossless GGUF header reading/writing plus MTP grafting.
//!
//! The header parser keeps every metadata value as raw bytes so an untouched
//! key round-trips byte-for-byte; only the keys the graft must change are
//! substituted. Tensor data is streamed in bounded slices, never loaded whole.

use anyhow::{Context, Result, bail};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

const MAX_STRING: u64 = 64 * 1024 * 1024;
const MAX_ARRAY: u64 = 1 << 28;
const MAX_BLOB: usize = 256 * 1024 * 1024;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_KV: u64 = 1 << 20;
const MAX_DIMS: u32 = 16;

const GGUF_UINT8: u32 = 0;
const GGUF_INT8: u32 = 1;
const GGUF_UINT16: u32 = 2;
const GGUF_INT16: u32 = 3;
const GGUF_UINT32: u32 = 4;
const GGUF_INT32: u32 = 5;
const GGUF_FLOAT32: u32 = 6;
const GGUF_BOOL: u32 = 7;
const GGUF_STRING: u32 = 8;
const GGUF_ARRAY: u32 = 9;
const GGUF_UINT64: u32 = 10;
const GGUF_INT64: u32 = 11;
const GGUF_FLOAT64: u32 = 12;

fn scalar_size(vt: u32) -> Result<usize> {
    Ok(match vt {
        GGUF_UINT8 | GGUF_INT8 | GGUF_BOOL => 1,
        GGUF_UINT16 | GGUF_INT16 => 2,
        GGUF_UINT32 | GGUF_INT32 | GGUF_FLOAT32 => 4,
        GGUF_UINT64 | GGUF_INT64 | GGUF_FLOAT64 => 8,
        _ => bail!("unsupported GGUF scalar type {vt}"),
    })
}

fn read_exact_n(r: &mut impl Read, n: usize) -> Result<Vec<u8>> {
    if n > MAX_BLOB {
        bail!("refusing to read {n} bytes (exceeds {MAX_BLOB} byte cap)");
    }
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact_n(r, 4)?.try_into().unwrap()))
}
fn read_u64(r: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact_n(r, 8)?.try_into().unwrap()))
}
fn read_string(r: &mut impl Read) -> Result<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING {
        bail!("GGUF string length {len} exceeds cap; file misaligned?");
    }
    Ok(String::from_utf8_lossy(&read_exact_n(r, len as usize)?).into_owned())
}

/// Read a full metadata value and return its exact on-disk encoding, so it can
/// be written back verbatim. For strings this includes the length prefix; for
/// arrays it includes the element type, count, and every element.
fn capture_value(r: &mut impl Read, vt: u32) -> Result<Vec<u8>> {
    match vt {
        GGUF_STRING => {
            let len = read_u64(r)?;
            if len > MAX_STRING {
                bail!("GGUF string length {len} exceeds cap; file misaligned?");
            }
            let mut out = Vec::with_capacity(8 + len as usize);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&read_exact_n(r, len as usize)?);
            Ok(out)
        }
        GGUF_ARRAY => {
            let elem = read_u32(r)?;
            let count = read_u64(r)?;
            if count > MAX_ARRAY {
                bail!("GGUF array count {count} exceeds cap; file misaligned?");
            }
            let mut out = Vec::new();
            out.extend_from_slice(&elem.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            if elem == GGUF_STRING || elem == GGUF_ARRAY {
                for _ in 0..count {
                    out.extend_from_slice(&capture_value(r, elem)?);
                }
            } else {
                let size = scalar_size(elem)? as u64;
                let total = count
                    .checked_mul(size)
                    .context("GGUF array size overflow")?;
                out.extend_from_slice(&read_exact_n(r, total as usize)?);
            }
            Ok(out)
        }
        _ => read_exact_n(r, scalar_size(vt)?),
    }
}

fn write_u32(w: &mut impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn write_u64(w: &mut impl Write, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn write_string(w: &mut impl Write, s: &str) -> Result<()> {
    write_u64(w, s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

#[derive(Clone)]
struct MetaEntry {
    key: String,
    vtype: u32,
    raw: Vec<u8>,
}

struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

struct Header {
    meta: Vec<MetaEntry>,
    tensors: Vec<TensorEntry>,
    alignment: u64,
    data_start: u64,
    file_size: u64,
}

fn parse_header(path: &Path) -> Result<Header> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 16, file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("{} is not a GGUF file", path.display());
    }
    let version = read_u32(&mut r)?;
    if version != 3 {
        bail!("unsupported GGUF version {version} in {}", path.display());
    }
    let n_tensors = read_u64(&mut r)?;
    let n_kv = read_u64(&mut r)?;
    if n_tensors > MAX_TENSORS || n_kv > MAX_KV {
        bail!("implausible GGUF header in {}: {n_tensors} tensors, {n_kv} keys", path.display());
    }

    let mut meta = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = read_string(&mut r)?;
        let vtype = read_u32(&mut r)?;
        let raw = capture_value(&mut r, vtype)?;
        meta.push(MetaEntry { key, vtype, raw });
    }

    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        if n_dims > MAX_DIMS {
            bail!("tensor '{name}' has {n_dims} dimensions; file misaligned?");
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut r)?);
        }
        let ggml_type = read_u32(&mut r)?;
        let offset = read_u64(&mut r)?;
        tensors.push(TensorEntry { name, dims, ggml_type, offset });
    }

    let header_end = r.stream_position()?;
    let alignment = meta
        .iter()
        .find(|m| m.key == "general.alignment")
        .and_then(|m| match (m.vtype, m.raw.len()) {
            (GGUF_UINT32, 4) => Some(u32::from_le_bytes(m.raw[..4].try_into().ok()?) as u64),
            (GGUF_UINT64, 8) => Some(u64::from_le_bytes(m.raw[..8].try_into().ok()?)),
            _ => None,
        })
        .unwrap_or(32);
    let data_start = header_end.div_ceil(alignment) * alignment;

    Ok(Header { meta, tensors, alignment, data_start, file_size })
}

fn meta_string(h: &Header, key: &str) -> Result<String> {
    let m = h
        .meta
        .iter()
        .find(|m| m.key == key)
        .with_context(|| format!("missing metadata key '{key}'"))?;
    if m.vtype != GGUF_STRING || m.raw.len() < 8 {
        bail!("metadata key '{key}' is not a string");
    }
    let len = u64::from_le_bytes(m.raw[..8].try_into().unwrap()) as usize;
    if 8 + len != m.raw.len() {
        bail!("metadata key '{key}' has inconsistent length");
    }
    Ok(String::from_utf8_lossy(&m.raw[8..]).into_owned())
}

fn meta_u64(h: &Header, key: &str) -> Result<u64> {
    let m = h
        .meta
        .iter()
        .find(|m| m.key == key)
        .with_context(|| format!("missing metadata key '{key}'"))?;
    match (m.vtype, m.raw.len()) {
        (GGUF_UINT32, 4) => Ok(u32::from_le_bytes(m.raw[..4].try_into().unwrap()) as u64),
        (GGUF_UINT64, 8) => Ok(u64::from_le_bytes(m.raw[..8].try_into().unwrap())),
        _ => bail!("metadata key '{key}' is not an unsigned integer"),
    }
}

fn on_disk_sizes(h: &Header) -> Vec<u64> {
    let mut sizes = Vec::with_capacity(h.tensors.len());
    for i in 0..h.tensors.len() {
        let this = h.data_start + h.tensors[i].offset;
        let next = if i + 1 < h.tensors.len() {
            h.data_start + h.tensors[i + 1].offset
        } else {
            h.file_size
        };
        sizes.push(next - this);
    }
    sizes
}

fn copy_n(src: &mut impl Read, dst: &mut impl Write, mut n: u64) -> Result<()> {
    let mut buf = vec![0u8; 8 << 20];
    while n > 0 {
        let chunk = n.min(buf.len() as u64) as usize;
        src.read_exact(&mut buf[..chunk])?;
        dst.write_all(&buf[..chunk])?;
        n -= chunk as u64;
    }
    Ok(())
}

pub struct GraftReport {
    pub grafted_tensors: usize,
    pub total_tensors: usize,
    pub block_count: u64,
    pub nextn_layers: u64,
    pub output_bytes: u64,
}

/// Transplant the MTP block (`blk.N.*`) from `donor` into `target`, writing the
/// mixed-quantization result to `output`.
pub fn graft_mtp(
    target: &Path,
    donor: &Path,
    output: &Path,
    mut progress: impl FnMut(u64, u64),
) -> Result<GraftReport> {
    if output.exists() {
        bail!("output {} already exists; remove it first", output.display());
    }
    if output == target || output == donor {
        bail!("output must not overwrite the target or donor");
    }

    let t = parse_header(target)?;
    let d = parse_header(donor)?;

    let arch = meta_string(&t, "general.architecture")?;
    let d_arch = meta_string(&d, "general.architecture")?;
    if d_arch != arch {
        bail!("architecture mismatch: target is {arch}, donor is {d_arch}");
    }

    let t_block = meta_u64(&t, &format!("{arch}.block_count"))?;
    let d_block = meta_u64(&d, &format!("{arch}.block_count"))?;
    let d_nextn = meta_u64(&d, &format!("{arch}.nextn_predict_layers"))?;
    if d_nextn == 0 {
        bail!("donor declares no nextn_predict_layers");
    }

    let prefix = format!("blk.{t_block}.");
    let extra: Vec<&TensorEntry> = d
        .tensors
        .iter()
        .filter(|x| x.name.starts_with(&prefix))
        .collect();
    if extra.is_empty() {
        bail!(
            "donor has no tensors with prefix '{prefix}' (donor block_count {d_block}, target {t_block})"
        );
    }
    if !extra.iter().any(|x| x.name.contains(".nextn.")) {
        bail!("donor tensors with prefix '{prefix}' are not an MTP block (no .nextn. tensors)");
    }
    for x in &extra {
        if t.tensors.iter().any(|y| y.name == x.name) {
            bail!("target already contains tensor '{}'", x.name);
        }
    }

    let t_sizes = on_disk_sizes(&t);
    let d_sizes = on_disk_sizes(&d);
    let d_map: HashMap<&str, (u64, u64)> = d
        .tensors
        .iter()
        .zip(&d_sizes)
        .map(|(x, s)| (x.name.as_str(), (d.data_start + x.offset, *s)))
        .collect();

    // Metadata: everything from the target except the two keys the graft
    // controls, then those two keys verbatim from the donor.
    let block_key = format!("{arch}.block_count");
    let nextn_key = format!("{arch}.nextn_predict_layers");
    let mut out_meta: Vec<MetaEntry> = t
        .meta
        .iter()
        .filter(|m| m.key != block_key && m.key != nextn_key)
        .cloned()
        .collect();
    for key in [&block_key, &nextn_key] {
        let m = d
            .meta
            .iter()
            .find(|m| &m.key == key)
            .with_context(|| format!("donor lacks metadata key '{key}'"))?;
        out_meta.push(m.clone());
    }

    // Tensor list and output offsets.
    let mut out_tensors: Vec<(&TensorEntry, u64)> = Vec::with_capacity(t.tensors.len() + extra.len());
    for (x, size) in t.tensors.iter().zip(&t_sizes) {
        out_tensors.push((x, *size));
    }
    for x in &extra {
        out_tensors.push((x, d_map[x.name.as_str()].1));
    }
    let mut offsets = Vec::with_capacity(out_tensors.len());
    let mut cur = 0u64;
    for (_, size) in &out_tensors {
        offsets.push(cur);
        cur += size;
    }
    let total_bytes = cur;

    let out_file = File::create(output).with_context(|| format!("create {}", output.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, out_file);
    w.write_all(b"GGUF")?;
    write_u32(&mut w, 3)?;
    write_u64(&mut w, out_tensors.len() as u64)?;
    write_u64(&mut w, out_meta.len() as u64)?;

    for m in &out_meta {
        write_string(&mut w, &m.key)?;
        write_u32(&mut w, m.vtype)?;
        w.write_all(&m.raw)?;
    }
    for (i, (x, _)) in out_tensors.iter().enumerate() {
        write_string(&mut w, &x.name)?;
        write_u32(&mut w, x.dims.len() as u32)?;
        for d in &x.dims {
            write_u64(&mut w, *d)?;
        }
        write_u32(&mut w, x.ggml_type)?;
        write_u64(&mut w, offsets[i])?;
    }

    let pos = w.stream_position()?;
    let pad = (t.alignment - (pos % t.alignment)) % t.alignment;
    if pad > 0 {
        w.write_all(&vec![0u8; pad as usize])?;
    }

    // Stream-copy tensor data.
    let mut t_in = File::open(target)?;
    let mut d_in = File::open(donor)?;
    let mut copied = 0u64;
    for (i, (x, size)) in out_tensors.iter().enumerate() {
        let (src, abs) = if i < t.tensors.len() {
            (&mut t_in, t.data_start + t.tensors[i].offset)
        } else {
            (&mut d_in, d_map[x.name.as_str()].0)
        };
        src.seek(SeekFrom::Start(abs))?;
        copy_n(src, &mut w, *size)?;
        copied += size;
        progress(copied, total_bytes);
    }
    w.flush()?;

    Ok(GraftReport {
        grafted_tensors: extra.len(),
        total_tensors: out_tensors.len(),
        block_count: d_block,
        nextn_layers: d_nextn,
        output_bytes: total_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture_bytes(vt: u32, input: &[u8]) -> Result<Vec<u8>> {
        let mut cursor = std::io::Cursor::new(input);
        capture_value(&mut cursor, vt)
    }

    #[test]
    fn capture_value_round_trips_scalars() {
        let u32v = 42u32.to_le_bytes().to_vec();
        assert_eq!(capture_bytes(GGUF_UINT32, &u32v).unwrap(), u32v);

        let i16v = (-1234i16).to_le_bytes().to_vec();
        assert_eq!(capture_bytes(GGUF_INT16, &i16v).unwrap(), i16v);

        let f64v = 2.718281828459045f64.to_le_bytes().to_vec();
        assert_eq!(capture_bytes(GGUF_FLOAT64, &f64v).unwrap(), f64v);
    }

    #[test]
    fn capture_value_round_trips_string() {
        let mut input = Vec::new();
        input.extend_from_slice(&3u64.to_le_bytes());
        input.extend_from_slice(b"abc");
        assert_eq!(capture_bytes(GGUF_STRING, &input).unwrap(), input);
    }

    #[test]
    fn capture_value_round_trips_arrays() {
        let mut arr = Vec::new();
        arr.extend_from_slice(&GGUF_UINT32.to_le_bytes());
        arr.extend_from_slice(&3u64.to_le_bytes());
        for v in [1u32, 2, 3] {
            arr.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(capture_bytes(GGUF_ARRAY, &arr).unwrap(), arr);

        let mut sarr = Vec::new();
        sarr.extend_from_slice(&GGUF_STRING.to_le_bytes());
        sarr.extend_from_slice(&2u64.to_le_bytes());
        for s in ["a", "bb"] {
            sarr.extend_from_slice(&(s.len() as u64).to_le_bytes());
            sarr.extend_from_slice(s.as_bytes());
        }
        assert_eq!(capture_bytes(GGUF_ARRAY, &sarr).unwrap(), sarr);
    }
}
