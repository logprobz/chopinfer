//! Snapshot retained head operands after replay, never inside graph capture.
use super::Instance;
use crate::dspark::{device::slice, execution::DraftInput, state::BLOCK_ROWS};
use serde_json::{Value, json};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::Path;

fn write_new(root: &Path, name: &str, bytes: &[u8]) -> Result<Value, String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(root.join(name))
        .map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    Ok(json!({"file": name, "bytes": bytes.len(), "sha256": kernel_build::digest(bytes)}))
}

impl Instance {
    pub(super) fn capture_draft_head(&mut self, inputs: &[DraftInput]) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let rows = inputs.len() as u32 * BLOCK_ROWS;
        let Some(capture) = device.head_capture.as_ref().filter(|c| c.pending(rows)) else {
            return Ok(());
        };
        let root = capture.root.clone();
        let directory = root.join(format!("rows-{rows}"));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .map_err(|e| format!("DSpark head capture: {e}"))?;
        let hidden = device.shape.geometry.shape.hidden;
        let vocab = device.shape.vocab_rows();
        let mut weights = capture.weights.clone();
        let mut ranks = Vec::new();
        for (rank, (buffers, stored)) in self.buffers.iter().zip(&device.ranks).enumerate() {
            if capture.weights.is_empty() {
                let span = slice(
                    buffers.view("head.weight")?,
                    0,
                    u64::from(vocab) * u64::from(hidden) * 2,
                )?;
                let bytes = self.read_capture_spans(rank, &[span])?;
                weights.push(write_new(&root, &format!("weight-r{rank}.bf16"), &bytes)?);
            }
            // The Markov positions read block_logits and use their own
            // position scratch. They leave both head operands intact.
            let activation = slice(
                buffers.scratch.normed_wide,
                0,
                u64::from(rows) * u64::from(hidden) * 4,
            )?;
            let output = slice(
                stored.block_logits,
                0,
                u64::from(rows) * u64::from(vocab) * 4,
            )?;
            let activation = self.read_capture_spans(rank, &[activation])?;
            let output = self.read_capture_spans(rank, &[output])?;
            ranks.push(json!({
                "rank": rank,
                "activation": write_new(&directory, &format!("activation-r{rank}.f32"), &activation)?,
                "output": write_new(&directory, &format!("output-r{rank}.f32"), &output)?,
                "weight": weights[rank],
            }));
        }
        let manifest = json!({
            "schema": "chopinfer.dspark-head-capture.v1",
            "geometry": device.shape.geometry.id,
            "rows": rows, "n": vocab, "k": hidden,
            "outside_graph_capture": true,
            "items": inputs.iter().map(|input| json!({
                "request": input.block.request,
                "position": input.block.main_rows,
                "bonus": input.bonus,
            })).collect::<Vec<_>>(),
            "ranks": ranks,
        });
        write_new(
            &directory,
            "capture.json",
            &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
        )?;
        let capture = self
            .dspark
            .as_mut()
            .and_then(|d| d.head_capture.as_mut())
            .expect("capture configured");
        capture.weights = weights;
        capture.captured.insert(rows);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    #[test]
    fn snapshot_files_are_private_and_never_replace_existing_entries() {
        let root = std::env::temp_dir().join(format!("dspark-head-files-{}", std::process::id()));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        let record = write_new(&root, "input.f32", &[1, 2, 3, 4]).unwrap();
        assert_eq!(record["bytes"], 4);
        assert_eq!(record["sha256"], kernel_build::digest(&[1, 2, 3, 4]));
        assert_eq!(
            std::fs::metadata(root.join("input.f32"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert!(write_new(&root, "input.f32", &[5]).is_err());
        symlink(root.join("input.f32"), root.join("link.f32")).unwrap();
        assert!(write_new(&root, "link.f32", &[5]).is_err());
        assert_eq!(std::fs::read(root.join("input.f32")).unwrap(), [1, 2, 3, 4]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
