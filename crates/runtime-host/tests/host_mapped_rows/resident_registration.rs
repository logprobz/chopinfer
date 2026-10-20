use super::*;

#[test]
fn default_cache_and_explicit_zero_keep_the_legacy_device_journal() {
    const CHILD: &str = "CHOPINFER_ROW_SET_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let mut journals = Vec::new();
        for enabled in [true, false] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command.args([
                "--exact",
                "resident_registration::default_cache_and_explicit_zero_keep_the_legacy_device_journal",
                "--nocapture",
            ]);
            command
                .env(CHILD, "1")
                .env("CHOPINFER_MAPPED_HOT_ROWS_SET_BYTES", "1048576")
                .env("CHOPINFER_MAPPED_HOT_ROWS_BYTES", "2097152")
                .env("CHOPINFER_MAPPED_READ_AHEAD", "0");
            if enabled {
                command.env_remove("CHOPINFER_MAPPED_HOT_ROWS");
            } else {
                command.env("CHOPINFER_MAPPED_HOT_ROWS", "0");
            }
            let result = command.output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let text = String::from_utf8(result.stdout).unwrap();
            let journal = text
                .lines()
                .find_map(|line| {
                    line.split_once("ROW_SET_DEVICE_JOURNAL ")
                        .map(|(_, row)| row)
                })
                .expect("executed child journal");
            journals.push(journal.to_owned());
            let diagnostics = String::from_utf8(result.stderr).unwrap();
            assert_eq!(diagnostics.contains("\"phase\":\"ready\""), enabled);
            assert!(
                !diagnostics.contains("\"phase\":\"uncached\""),
                "{diagnostics}"
            );
        }
        assert_eq!(journals[0], journals[1]);
        return;
    }
    let root = temp("resident-registration");
    write_checkpoint(&root.0);
    let image = planned(&root.0, vec![plan(1, 0)]).unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let image = host.open_compiled_image(ROLE.as_bytes()).unwrap();
    let snapshot = || {
        backend.with_device(|device| {
            serde_json::json!([
                device.live_object_count(),
                device.synchronizations(),
                device.host_to_device_copy_bytes(),
                device.refused_launches().len(),
                device.custom_operations().len(),
            ])
        })
    };
    let before = snapshot();
    let planes = [(b"codes".to_vec(), 8), (b"scales".to_vec(), 1)].map(|(view_id, row_bytes)| {
        runtime_bundle_abi::MappedRowPlaneV1 {
            image,
            rank: 0,
            view_id,
            row_bytes,
        }
    });
    host.register_mapped_row_set(&planes).unwrap();
    host.register_mapped_row_set(&planes).unwrap();
    assert_eq!(
        snapshot(),
        before,
        "registration must allocate no device or pinned staging"
    );
    assert_eq!(backend.host_memory_v1().gather_staging_bytes, 0);
    for bad in [
        runtime_bundle_abi::MappedRowPlaneV1 {
            rank: 1,
            ..planes[0].clone()
        },
        runtime_bundle_abi::MappedRowPlaneV1 {
            row_bytes: 7,
            ..planes[0].clone()
        },
    ] {
        assert_eq!(
            host.register_mapped_row_set(&[bad]),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
        );
    }
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let buffer = host.allocate_device_buffer(pool, stream, 64, 64).unwrap();
    let rows = [7_u32, 0, 999, 7];
    let mut bytes = Vec::new();
    for plane in &planes {
        let destination = DeviceBufferSpanV1::new(buffer, 0, plane.row_bytes * rows.len() as u64);
        host.gather_mapped_rows(
            stream,
            destination,
            MappedRowGatherV1 {
                image,
                rank: 0,
                view_id: &plane.view_id,
                row_bytes: plane.row_bytes,
                rows: &rows,
            },
        )
        .unwrap();
        let mut actual = vec![0; destination.length as usize];
        host.copy_device_to_host(stream, &mut actual, destination)
            .unwrap();
        let source = if plane.row_bytes == 8 {
            codes()
        } else {
            scales()
        };
        let width = plane.row_bytes as usize;
        let expected = rows
            .iter()
            .flat_map(|&row| source[row as usize * width..][..width].to_vec())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        bytes.extend(actual);
    }
    host.release_device_buffer(buffer).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
    host.release_compiled_image(image).unwrap();
    println!(
        "ROW_SET_DEVICE_JOURNAL {}",
        serde_json::json!({"device":snapshot(),"bytes":bytes})
    );
    let flushed = runtime_engine::logging::flush();
    assert!(flushed.flushed && flushed.dropped_lines == 0);
}
