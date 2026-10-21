use super::*;
use runtime_bundle_abi::weight_plan::{CheckpointFileV1, NamedBytesV1};

// Tests retain their small requests; production creates one request at a time.
fn prepare<T>(
    cache: Option<&Cache>,
    requests: &[WeightPlanRequestV1],
    generate: impl FnMut(&WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String>,
    validate: impl FnMut(&[WeightPlanV1]) -> Result<T, String>,
) -> Result<Option<(T, usize, usize)>, String> {
    super::prepare(
        cache,
        requests.len(),
        |rank| requests[rank].clone(),
        generate,
        validate,
    )
}

#[test]
fn every_request_input_changes_the_cache_key() {
    let base = WeightPlanRequestV1 {
        image_role: "weights".into(),
        ranks: 2,
        rank: 0,
        files: vec![CheckpointFileV1 {
            name: "weights.safetensors".into(),
            byte_length: 100,
            data_offset: 8,
            header: vec![1],
        }],
        documents: vec![NamedBytesV1 {
            name: "config.json".into(),
            bytes: vec![2],
        }],
        components: vec![NamedBytesV1 {
            name: "metadata".into(),
            bytes: vec![3],
        }],
        host_views: vec!["table".into()],
        ..Default::default()
    };
    let changes: Vec<fn(&mut WeightPlanRequestV1)> = vec![
        |r| r.image_role.push('x'),
        |r| r.ranks += 1,
        |r| r.rank += 1,
        |r| r.checkpoint_identity[0] += 1,
        |r| r.implementation.bundle_module[0] += 1,
        |r| r.implementation.shared_transforms[0] += 1,
        |r| r.files[0].name.push('x'),
        |r| r.files[0].byte_length += 1,
        |r| r.files[0].data_offset += 1,
        |r| r.files[0].header.push(2),
        |r| r.documents[0].name.push('x'),
        |r| r.documents[0].bytes.push(3),
        |r| r.components[0].name.push('x'),
        |r| r.components[0].bytes.push(4),
        |r| r.host_views[0].push('x'),
    ];
    for change in changes {
        let mut request = base.clone();
        change(&mut request);
        assert_ne!(key(&base), key(&request));
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "weight-plan-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn cache(&self) -> Cache {
            Cache::open(&self.0.join("cache")).unwrap()
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn request() -> WeightPlanRequestV1 {
        WeightPlanRequestV1 {
            image_role: "weights".into(),
            ranks: 1,
            ..Default::default()
        }
    }
    fn plan() -> WeightPlanV1 {
        WeightPlanV1 {
            image_role: "weights".into(),
            ranks: 1,
            rank: 0,
            alignment: 256,
            arena_bytes: 256,
            views: vec![],
            components: vec![],
            records: vec![],
        }
    }

    #[test]
    fn hits_preserve_plans_and_failed_validation_replans() {
        let temp = Temp::new();
        let cache = temp.cache();
        let request = request();
        let mut calls = 0;
        let generated = prepare(
            Some(&cache),
            std::slice::from_ref(&request),
            |_| {
                calls += 1;
                Ok(Some(plan()))
            },
            |plans| Ok(plans.to_vec()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(generated.2, 0);
        let loaded = prepare(
            Some(&cache),
            std::slice::from_ref(&request),
            |_| panic!("cache miss"),
            |plans| Ok(plans.to_vec()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(loaded.0, generated.0);
        assert_eq!(loaded.2, 1);
        let mut validations = 0;
        let replanned = prepare(
            Some(&cache),
            &[request],
            |_| {
                calls += 1;
                Ok(Some(plan()))
            },
            |plans| {
                validations += 1;
                if validations == 1 {
                    Err("source changed".into())
                } else {
                    Ok(plans.to_vec())
                }
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!((calls, validations, replanned.2), (2, 2, 0));
    }

    #[test]
    fn internally_valid_plans_for_another_role_or_rank_are_cache_misses() {
        let temp = Temp::new();
        let cache = temp.cache();
        let request = request();
        let mutations: Vec<fn(&mut WeightPlanV1)> = vec![
            |plan| plan.image_role = "another-role".into(),
            |plan| plan.ranks = 2,
            |plan| {
                plan.ranks = 2;
                plan.rank = 1;
            },
        ];
        for mutate in mutations {
            let mut wrong = plan();
            mutate(&mut wrong);
            cache.write(&key(&request), &wrong.encode());
            assert!(
                cache.read(&key(&request)).is_some(),
                "envelope and codec remain valid"
            );
            let mut calls = 0;
            let result = prepare(
                Some(&cache),
                std::slice::from_ref(&request),
                |_| {
                    calls += 1;
                    Ok(Some(plan()))
                },
                |plans| Ok(plans.to_vec()),
            )
            .unwrap()
            .unwrap();
            assert_eq!(calls, 1);
            assert_eq!(result.2, 0);
            assert_eq!(result.0, vec![plan()]);
        }
    }

    #[test]
    fn cached_multirank_images_keep_bytes_and_revalidate_sources() {
        use crate::direct::{
            DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
        };
        use kernel_core::weight_transform::WeightTransform;
        use runtime_bundle_abi::weight_plan::{RecordSourceV1, WeightRecordV1};
        use std::sync::Arc;
        let temp = Temp::new();
        let root = temp.0.join("checkpoint");
        fs::create_dir(&root).unwrap();
        let header = br#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[3, 5, 7, 11]);
        fs::write(root.join("model.safetensors"), bytes).unwrap();
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root).unwrap());
        let implementation = implementation_v1([0; 32]);
        let requests: Vec<_> = (0..2)
            .map(|rank| checkpoint.request("weights", 2, rank, implementation, vec![], vec![]))
            .collect();
        let plans: Vec<_> = (0..2)
            .map(|rank| WeightPlanV1 {
                ranks: 2,
                rank,
                records: vec![WeightRecordV1 {
                    arena_offset: 0,
                    byte_length: 4,
                    transform: WeightTransform::Copy.id(),
                    params: WeightTransform::Copy.encode_params().unwrap(),
                    sources: vec![RecordSourceV1 {
                        file: 0,
                        tensor: "x".into(),
                    }],
                    axis: None,
                    ranges: vec![],
                    stored_shape: vec![4],
                }],
                ..plan()
            })
            .collect();
        for plan in &plans {
            assert!(super::super::encoded_bound(plan) >= plan.encode().len());
        }
        let validate = |plans: &[WeightPlanV1]| {
            PlannedImageV1::new_with_range_tracking(
                checkpoint.clone(),
                plans,
                None,
                implementation,
                DirectLoadOptionsV1 {
                    threads_per_rank: 2,
                    read_ahead: false,
                    direct_reads: false,
                    shared_bytes: 1024,
                },
                false,
            )
        };
        let cache = temp.cache();
        for expected_hits in [0, 2] {
            let mut generated = 0;
            let (image, records, hits) = prepare(
                Some(&cache),
                &requests,
                |request| {
                    generated += 1;
                    Ok(Some(plans[request.rank as usize].clone()))
                },
                validate,
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                (records, hits, generated),
                (2, expected_hits, 2 - expected_hits)
            );
            let mut got = vec![0xaa; 512];
            image.read(0, &mut got).unwrap();
            let mut expected = vec![0; 512];
            expected[..4].copy_from_slice(&[3, 5, 7, 11]);
            expected[256..260].copy_from_slice(&[3, 5, 7, 11]);
            assert_eq!(got, expected);
        }
        let mut invalid = plans[0].clone();
        invalid.records[0].sources[0].tensor = "missing".into();
        cache.write(&key(&requests[0]), &invalid.encode());
        let mut generated = 0;
        let (_, _, hits) = prepare(
            Some(&cache),
            &requests,
            |request| {
                generated += 1;
                Ok(Some(plans[request.rank as usize].clone()))
            },
            validate,
        )
        .unwrap()
        .unwrap();
        assert_eq!((hits, generated), (0, 2));
    }

    #[test]
    fn corrupt_truncated_wrong_key_and_oversized_entries_replan() {
        let temp = Temp::new();
        let cache = temp.cache();
        let request = request();
        let key = key(&request);
        for corruption in 0..5 {
            cache.write(&key, &plan().encode());
            let path = cache.path(&key);
            let mut bytes = fs::read(&path).unwrap();
            match corruption {
                0 => {
                    bytes[0] ^= 1;
                    fs::write(&path, bytes).unwrap();
                }
                1 => {
                    bytes[8] ^= 1;
                    fs::write(&path, bytes).unwrap();
                }
                2 => {
                    let last = bytes.len() - 1;
                    bytes[last] ^= 1;
                    fs::write(&path, bytes).unwrap();
                }
                3 => fs::write(&path, &bytes[..40]).unwrap(),
                4 => fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_len((256 << 20) + 1)
                    .unwrap(),
                _ => unreachable!(),
            }
            let mut calls = 0;
            let loaded = prepare(
                Some(&cache),
                std::slice::from_ref(&request),
                |_| {
                    calls += 1;
                    Ok(Some(plan()))
                },
                |plans| Ok(plans.to_vec()),
            )
            .unwrap()
            .unwrap();
            assert_eq!((calls, loaded.2), (1, 0));
        }
        // Valid envelope and digest with invalid encoded contents is a miss.
        cache.write(&key, b"not a plan");
        assert!(cache.read(&key).is_none());
    }

    #[test]
    fn private_paths_refuse_symlinks_hardlinks_and_open_permissions() {
        let temp = Temp::new();
        let cache = temp.cache();
        let key = key(&request());
        cache.write(&key, &plan().encode());
        let path = cache.path(&key);
        let other = temp.0.join("other");
        fs::hard_link(&path, &other).unwrap();
        assert!(cache.read(&key).is_none());
        fs::remove_file(&other).unwrap();
        fs::rename(&path, &other).unwrap();
        symlink(&other, &path).unwrap();
        assert!(cache.read(&key).is_none());
        assert!(cache.publish(&key, &plan().encode(), 1024, 3).is_err());
        symlink(temp.0.join("cache"), temp.0.join("link")).unwrap();
        assert!(Cache::open(&temp.0.join("link")).is_err());
        assert!(Cache::open(&temp.0.join("link/child")).is_err());
        fs::set_permissions(temp.0.join("cache"), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(Cache::open(&temp.0.join("cache")).is_err());
    }

    #[test]
    fn concurrent_publication_is_atomic_and_respects_the_budget() {
        let temp = Temp::new();
        let key = key(&request());
        let bytes = plan().encode();
        let cache = temp.cache();
        cache.write(&key, &bytes);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let cache = temp.cache();
                let (key, bytes) = (&key, &bytes);
                scope.spawn(move || {
                    for _ in 0..10 {
                        cache.write(key, bytes);
                        // Replacement may unlink the opened inode before its
                        // metadata check. That safe miss falls back to planning.
                        if let Some(loaded) = cache.read(key) {
                            assert_eq!(loaded, plan());
                        }
                    }
                });
            }
        });
        assert_eq!(cache.read(&key).unwrap(), plan());
        let size = fs::metadata(cache.path(&key)).unwrap().len();
        for rank in 1..6 {
            let mut request = request();
            request.rank = rank;
            cache
                .publish(&super::key(&request), &bytes, size * 2, 2)
                .unwrap();
            let files: Vec<_> = fs::read_dir(temp.0.join("cache"))
                .unwrap()
                .map(Result::unwrap)
                .collect();
            assert!(files.len() <= 2);
            assert!(
                files
                    .iter()
                    .map(|file| file.metadata().unwrap().len())
                    .sum::<u64>()
                    <= size * 2
            );
        }
        assert!(cache.publish(&key, &bytes, size - 1, 2).is_err());
    }

    #[test]
    fn rejected_generated_plans_are_not_published() {
        let temp = Temp::new();
        let cache = temp.cache();
        let request = request();
        let result = prepare(
            Some(&cache),
            std::slice::from_ref(&request),
            |_| Ok(Some(plan())),
            |_| Err::<(), _>("invalid plan".into()),
        );
        assert!(result.is_err());
        assert!(cache.read(&key(&request)).is_none());
        assert!(
            prepare::<()>(None, &[request], |_| Ok(None), |_| unreachable!())
                .unwrap()
                .is_none()
        );
    }
}
