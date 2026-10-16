use super::*;
use serde_json::json;

async fn start(backend: Arc<dyn GenerationBackend>, config: ServerConfig) -> TestServerV1 {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(serve(
        listener,
        backend,
        Arc::new(FakeFrontendV1::new()),
        config,
    ));
    TestServerV1 { address }
}
fn stored_config(entries: usize) -> ServerConfig {
    ServerConfig::new(8, 8, 65536, 4096)
        .unwrap()
        .with_responses_store(entries, 1_048_576)
}
async fn create(server: &TestServerV1, body: Value) -> ResponseV1 {
    server
        .request(&post("/v1/responses", &body.to_string()))
        .await
}
fn events(response: &ResponseV1) -> Vec<Value> {
    assert_eq!(response.status, 200, "{}", response.body);
    assert!(!response.body.contains("[DONE]"));
    let names: Vec<_> = response
        .body
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect();
    let values: Vec<Value> = response
        .body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(names.len(), values.len());
    for (i, (name, value)) in names.iter().zip(&values).enumerate() {
        assert_eq!(value["type"], *name);
        assert_eq!(value["sequence_number"], i);
    }
    values
}
fn tool() -> Value {
    json!({"type":"function","name":"fixture.lookup","parameters":{"type":"object"}})
}
fn tool_events() -> Vec<GenerationEvent> {
    vec![
        GenerationEvent::ToolCallDeltas(vec![runtime_server::ToolCallDelta {
            index: 0,
            name: Some("fixture.lookup".to_owned()),
            arguments: "{".to_owned(),
        }]),
        GenerationEvent::ToolCallDeltas(vec![runtime_server::ToolCallDelta {
            index: 0,
            name: None,
            arguments: "}".to_owned(),
        }]),
        GenerationEvent::ToolCalls {
            calls: vec![BackendToolCall {
                name: "fixture.lookup".to_owned(),
                arguments: "{}".to_owned(),
            }],
            generated_token_count: 1,
        },
        GenerationEvent::Finished {
            reason: FinishReason::ToolCalls,
            usage: TokenUsage {
                prompt_tokens: 4,
                completion_tokens: 1,
                cached_tokens: None,
            },
        },
    ]
}

#[tokio::test]
async fn text_request_uses_chat_preparation_and_usage() {
    let backend = ScriptedBackendV1::new(vec![content("hello"), finished()]);
    let server = TestServerV1::start(backend.clone()).await;
    let response = create(&server,json!({"model":ALIAS,"input":"question","instructions":"be brief","max_output_tokens":5,"temperature":0.2,"top_p":0.8,"metadata":{"case":"text"}})).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let value = response.json();
    assert_eq!(value["object"], "response");
    assert!(value["top_logprobs"].is_null());
    assert_eq!(value["status"], "completed");
    assert!(value["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(value["output"][0]["content"][0]["text"], "hello");
    assert_eq!(value["usage"]["input_tokens"], 4);
    assert_eq!(value["usage"]["output_tokens"], 1);
    assert_eq!(value["usage"]["total_tokens"], 5);
    assert_eq!(value["metadata"]["case"], "text");
    let prompt = &backend.prompts()[0];
    assert_eq!(prompt.max_new_tokens, 5);
    assert_eq!(prompt.sampling.temperature(), 0.2);
    assert!(
        matches!(&prompt.prompt,PreparedPrompt::RenderedChat(text) if text == "system:be brief|user:question")
    );
    assert_eq!(
        server
            .request(&get(&format!(
                "/v1/responses/{}",
                value["id"].as_str().unwrap()
            )))
            .await
            .status,
        404
    );
}
#[tokio::test]
async fn stream_has_reference_text_event_order_and_stable_items() {
    let server =
        TestServerV1::start(ScriptedBackendV1::new(vec![content("hello"), finished()])).await;
    let response = create(&server, json!({"input":"x","stream":true})).await;
    let values = events(&response);
    let names: Vec<_> = values.iter().map(|v| v["type"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    assert_eq!(values[2]["item"]["id"], values[4]["item_id"]);
    assert_eq!(values[7]["item"], values[8]["response"]["output"][0]);
    assert_eq!(values[8]["response"]["usage"]["output_tokens"], 1);
}
#[tokio::test]
async fn tool_arguments_stream_and_round_trip_to_chat_history() {
    let backend = ScriptedBackendV1::new(tool_events());
    let server = TestServerV1::start(backend.clone()).await;
    let response = create(&server, json!({"input":"x","stream":true,"tools":[tool()]})).await;
    let values = events(&response);
    let names: Vec<_> = values.iter().map(|v| v["type"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    let call = values.last().unwrap()["response"]["output"][0].clone();
    assert_eq!(call["arguments"], "{}");
    let response=create(&server,json!({"input":[{"role":"user","content":"x"},call,{"type":"function_call_output","call_id":call["call_id"],"output":"result"}],"tools":[tool()]})).await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert!(
        matches!(&backend.prompts()[1].prompt,PreparedPrompt::RenderedChat(text) if text.contains("assistant:") && text.contains("tool:result"))
    );
}
#[tokio::test]
async fn reasoning_items_use_declared_field_and_are_counted_when_hidden() {
    for field in ["reasoning", "reasoning_content"] {
        for hidden in [false, true] {
            let backend = ScriptedBackendV1::new(vec![
                GenerationEvent::Reasoning(ReasoningContent {
                    delta: "think".to_owned(),
                    generated_token_count: 1,
                }),
                content("answer"),
                GenerationEvent::Finished {
                    reason: FinishReason::Stop,
                    usage: TokenUsage {
                        prompt_tokens: 4,
                        completion_tokens: 2,
                        cached_tokens: Some(2),
                    },
                },
            ]);
            let server = TestServerV1::start_with_frontend(
                backend,
                FakeFrontendV1::with_reasoning_field(field),
            )
            .await;
            let response=create(&server,json!({"input":"x","stream":true,"reasoning":{"effort":"high"},"include_reasoning":!hidden})).await;
            let values = events(&response);
            let result = &values.last().unwrap()["response"];
            assert_eq!(
                result["usage"]["output_tokens_details"]["reasoning_tokens"],
                1
            );
            assert_eq!(result["usage"]["input_tokens_details"]["cached_tokens"], 2);
            if hidden {
                assert_eq!(result["output"][0]["type"], "message");
            } else {
                assert_eq!(result["output"][0]["type"], "reasoning");
                assert_eq!(result["output"][0]["content"][0]["text"], "think");
                let names: Vec<_> = values.iter().map(|v| v["type"].as_str().unwrap()).collect();
                assert_eq!(
                    &names[2..8],
                    &[
                        "response.output_item.added",
                        "response.reasoning_part.added",
                        "response.reasoning_text.delta",
                        "response.reasoning_text.done",
                        "response.reasoning_part.done",
                        "response.output_item.done"
                    ]
                );
            }
        }
    }
}
#[tokio::test]
async fn structured_output_and_reasoning_history_reach_the_shared_frontend() {
    let backend = ScriptedBackendV1::new(vec![content("{}"), finished()]);
    let server = TestServerV1::start(backend.clone()).await;
    let response=create(&server,json!({"input":[{"type":"reasoning","content":[{"type":"reasoning_text","text":"prior"}],"summary":[]},{"role":"assistant","content":[{"type":"output_text","text":"answer"}]},{"role":"user","content":[{"type":"input_text","text":"next"}]}],"text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object","properties":{},"additionalProperties":false},"strict":true}}})).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let prompts = backend.prompts();
    assert!(prompts[0].constraint.is_some());
    assert!(
        matches!(&prompts[0].prompt,PreparedPrompt::RenderedChat(text) if text == "assistant:answer~prior|user:next")
    );
    assert_eq!(response.json()["text"]["format"]["type"], "json_schema");
}
#[tokio::test]
async fn invalid_requests_and_storage_errors_match_reference_statuses() {
    let backend = ScriptedBackendV1::new(vec![content("x"), finished()]);
    let server = TestServerV1::start(backend.clone()).await;
    for body in [
        json!({}),
        json!({"input":3}),
        json!({"input":"x","max_output_tokens":0}),
        json!({"input":"x","tools":[{"type":"web_search"}]}),
        json!({"input":"x","background":true}),
        json!({"input":"x","background":true,"store":false}),
        json!({"input":"x","reasoning":{"effort":"impossible"}}),
        json!({"input":[{"type":"function_call_output","call_id":"c","output":3}]}),
        json!({"input":"x","text":{"format":{"type":"invalid"}}}),
        json!({"input":"x","stream":"true"}),
        json!({"input":"x","unknown_control":true}),
        json!({"input":"x","truncation":"auto"}),
        json!({"input":"x","service_tier":"priority"}),
        json!({"input":"x","reasoning":{"summary":"auto"}}),
        json!({"input":"x","include":["reasoning.encrypted_content"]}),
        json!({"input":"x","prompt":{"id":"p"}}),
        json!({"input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]}),
    ] {
        let response = create(&server, body.clone()).await;
        assert_eq!(response.status, 400, "{body}: {}", response.body);
        assert!(response.json()["error"]["message"].is_string());
    }
    for body in [
        json!({"input":"x","model":"missing"}),
        json!({"input":"x","previous_response_id":"missing"}),
    ] {
        assert_eq!(create(&server, body).await.status, 404);
    }
    assert_eq!(
        server.request(&get("/v1/responses/missing")).await.status,
        404
    );
    let error = server
        .request(&post("/v1/responses/missing/cancel", "{}"))
        .await;
    assert_eq!(error.status, 404);
    assert_eq!(error.json()["error"]["type"], "invalid_request_error");
    assert_eq!(backend.submissions.load(Ordering::Relaxed), 0);
}
#[tokio::test]
async fn backend_errors_and_length_limits_have_terminal_contracts() {
    for streaming in [false, true] {
        let server = TestServerV1::start(ScriptedBackendV1::new(vec![GenerationEvent::Failed {
            message: "test failure".to_owned(),
            code: None,
        }]))
        .await;
        let response = create(&server, json!({"input":"x","stream":streaming})).await;
        if streaming {
            let values = events(&response);
            assert_eq!(values.last().unwrap()["type"], "error");
            assert!(!values.iter().any(|v| v["type"] == "response.completed"));
        } else {
            assert_eq!(response.status, 500);
        }
    }
    let server = TestServerV1::start(ScriptedBackendV1::new(vec![
        content("x"),
        GenerationEvent::Finished {
            reason: FinishReason::Length,
            usage: TokenUsage {
                prompt_tokens: 4,
                completion_tokens: 1,
                cached_tokens: None,
            },
        },
    ]))
    .await;
    let response = create(
        &server,
        json!({"input":"x","max_output_tokens":1,"stream":true}),
    )
    .await;
    let values = events(&response);
    let last = values.last().unwrap();
    assert_eq!(last["type"], "response.completed");
    assert_eq!(last["response"]["status"], "incomplete");
    assert_eq!(
        last["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
}
#[tokio::test]
async fn stored_history_replaces_instructions_and_evicts_completed_entries() {
    let backend = ScriptedBackendV1::new(vec![content("answer"), finished()]);
    let server = start(backend.clone(), stored_config(1)).await;
    let first = create(&server, json!({"input":"first","instructions":"old"}))
        .await
        .json();
    let id = first["id"].as_str().unwrap();
    let retrieved = server.request(&get(&format!("/v1/responses/{id}"))).await;
    assert_eq!(retrieved.json(), first);
    assert_eq!(
        server
            .request(&post(&format!("/v1/responses/{id}/cancel"), "{}"))
            .await
            .status,
        400
    );
    let next = create(
        &server,
        json!({"input":"second","instructions":"new","previous_response_id":id}),
    )
    .await;
    assert_eq!(next.status, 200, "{}", next.body);
    assert!(
        matches!(&backend.prompts()[1].prompt,PreparedPrompt::RenderedChat(text) if text == "system:new|user:first|assistant:answer|user:second")
    );
    assert_eq!(
        server
            .request(&get(&format!("/v1/responses/{id}")))
            .await
            .status,
        404
    );
    let last = create(&server, json!({"input":"not stored","store":false}))
        .await
        .json();
    assert_eq!(
        server
            .request(&get(&format!(
                "/v1/responses/{}",
                last["id"].as_str().unwrap()
            )))
            .await
            .status,
        404
    );
}

struct WaitingBackend {
    submissions: AtomicUsize,
    cancelled: tokio::sync::mpsc::UnboundedSender<()>,
}
impl GenerationBackend for WaitingBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }
    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        self.submissions.fetch_add(1, Ordering::Relaxed);
        let cancelled = self.cancelled.clone();
        Box::pin(async move {
            tokio::spawn(async move {
                request.cancellation.cancelled().await;
                drop(events);
                let _ = cancelled.send(());
            });
            Ok(())
        })
    }
}
#[tokio::test]
async fn background_cancellation_reaches_backend_and_preserves_cancelled_status() {
    let (cancelled, mut observed) = tokio::sync::mpsc::unbounded_channel();
    let server = start(
        Arc::new(WaitingBackend {
            cancelled,
            submissions: AtomicUsize::new(0),
        }),
        stored_config(1),
    )
    .await;
    let first = create(&server, json!({"input":"x","background":true})).await;
    assert_eq!(first.status, 200, "{}", first.body);
    let first = first.json();
    assert_eq!(first["status"], "queued");
    let id = first["id"].as_str().unwrap();
    assert_eq!(
        create(&server, json!({"input":"x","background":true}))
            .await
            .status,
        429
    );
    let cancel = server
        .request(&post(&format!("/v1/responses/{id}/cancel"), "{}"))
        .await;
    assert_eq!(cancel.status, 200);
    assert_eq!(cancel.json()["status"], "cancelled");
    timeout(Duration::from_secs(5), observed.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        server
            .request(&get(&format!("/v1/responses/{id}")))
            .await
            .json()["status"],
        "cancelled"
    );
}
#[tokio::test]
async fn responses_preserve_restricted_surface_and_output_bounds() {
    let server = start(
        ScriptedBackendV1::new(vec![content("too long"), finished()]),
        ServerConfig::new(1, 4, 4096, 4).unwrap(),
    )
    .await;
    assert_eq!(create(&server, json!({"input":"x"})).await.status, 500);
    let server = start(
        ScriptedBackendV1::new(vec![content("x"), finished()]),
        ServerConfig::default()
            .with_request_surface(runtime_server::RequestSurfaceV1::SingleTurnPlainChat),
    )
    .await;
    assert_eq!(create(&server, json!({"input":"x"})).await.status, 200);
    assert_eq!(
        create(&server, json!({"input":"x","tools":[tool()]}))
            .await
            .status,
        400
    );
}

#[tokio::test]
async fn logprobs_require_include_and_preserve_token_scores() {
    let backend = ScriptedBackendV1::new(vec![
        GenerationEvent::GeneratedTokenLogprobs(vec![runtime_server::GeneratedTokenLogprob {
            token_id: 7,
            token: "hello".to_owned(),
            bytes: b"hello".to_vec(),
            logprob: -0.25,
            top_logprobs: vec![runtime_server::TokenLogprob {
                token: "hello".to_owned(),
                bytes: b"hello".to_vec(),
                logprob: -0.25,
            }],
        }]),
        content("hello"),
        finished(),
    ]);
    let server = TestServerV1::start(backend).await;
    for stream in [false, true] {
        let response=create(&server,json!({"input":"x","stream":stream,"include":["message.output_text.logprobs"],"top_logprobs":1})).await;
        assert_eq!(response.status, 200, "{}", response.body);
        let value = if stream {
            let values = events(&response);
            let delta = values
                .iter()
                .find(|v| v["type"] == "response.output_text.delta")
                .unwrap();
            assert_eq!(delta["logprobs"][0]["logprob"], -0.25);
            values.last().unwrap()["response"].clone()
        } else {
            response.json()
        };
        assert_eq!(value["top_logprobs"], 1);
        assert_eq!(
            value["output"][0]["content"][0]["logprobs"][0]["token"],
            "hello"
        );
    }
    let backend = ScriptedBackendV1::new(vec![content("hello"), finished()]);
    let server = TestServerV1::start(backend.clone()).await;
    assert_eq!(
        create(&server, json!({"input":"x","top_logprobs":1}))
            .await
            .status,
        200
    );
    assert!(!backend.prompts()[0].logprobs.enabled);
}

#[tokio::test]
async fn stored_background_completes_and_small_store_refuses_without_leaking() {
    let server = start(
        ScriptedBackendV1::new(vec![content("done"), finished()]),
        stored_config(2),
    )
    .await;
    let initial = create(&server, json!({"input":"x","background":true}))
        .await
        .json();
    let path = format!("/v1/responses/{}", initial["id"].as_str().unwrap());
    timeout(Duration::from_secs(5), async {
        loop {
            let response = server.request(&get(&path)).await.json();
            if response["status"] == "completed" {
                assert_eq!(response["output"][0]["content"][0]["text"], "done");
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        server
            .request(&get(&format!("{path}?stream=true")))
            .await
            .status,
        400
    );
    let server = start(
        ScriptedBackendV1::new(vec![content("done"), finished()]),
        stored_config(2).with_responses_store(2, 10),
    )
    .await;
    assert_eq!(create(&server, json!({"input":"x"})).await.status, 400);
    assert_eq!(
        create(&server, json!({"input":"x","store":false}))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn disconnecting_a_response_stream_cancels_generation() {
    let (cancelled, mut observed) = tokio::sync::mpsc::unbounded_channel();
    let server = start(
        Arc::new(WaitingBackend {
            cancelled,
            submissions: AtomicUsize::new(0),
        }),
        ServerConfig::default(),
    )
    .await;
    let socket = TcpStream::connect(server.address).await.unwrap();
    let request = post("/v1/responses", r#"{"input":"x","stream":true}"#);
    socket.writable().await.unwrap();
    assert_eq!(socket.try_write(request.as_bytes()).unwrap(), request.len());
    timeout(Duration::from_secs(5), async {
        let mut seen = Vec::new();
        loop {
            socket.readable().await.unwrap();
            let mut bytes = [0; 4096];
            match socket.try_read(&mut bytes) {
                Ok(n) => seen.extend_from_slice(&bytes[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(e) => panic!("{e}"),
            }
            if String::from_utf8_lossy(&seen).contains("response.created") {
                break;
            }
        }
    })
    .await
    .unwrap();
    drop(socket);
    timeout(Duration::from_secs(5), observed.recv())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn sequential_calls_close_before_the_next_item_opens() {
    let backend = ScriptedBackendV1::new(vec![
        GenerationEvent::ToolCalls {
            calls: vec![
                BackendToolCall {
                    name: "fixture.lookup".to_owned(),
                    arguments: "{}".to_owned(),
                },
                BackendToolCall {
                    name: "fixture.lookup".to_owned(),
                    arguments: "{}".to_owned(),
                },
            ],
            generated_token_count: 1,
        },
        GenerationEvent::Finished {
            reason: FinishReason::ToolCalls,
            usage: TokenUsage {
                prompt_tokens: 4,
                completion_tokens: 1,
                cached_tokens: None,
            },
        },
    ]);
    let server = TestServerV1::start(backend).await;
    let response = create(&server, json!({"input":"x","tools":[tool()],"stream":true})).await;
    let values = events(&response);
    let names: Vec<_> = values
        .iter()
        .map(|value| value["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    assert_ne!(values[2]["item"]["call_id"], values[6]["item"]["call_id"]);
    assert_eq!(
        values.last().unwrap()["response"]["output"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn deferred_admission_errors_keep_their_http_status() {
    let server = TestServerV1::start(ScriptedBackendV1::new(vec![
        GenerationEvent::AdmissionFailed(AdmissionError::AtCapacity),
    ]))
    .await;
    assert_eq!(create(&server, json!({"input":"x"})).await.status, 429);
}

#[tokio::test]
async fn expanded_requests_use_the_minimum_intake_reservation_before_growing() {
    let server = start(
        ScriptedBackendV1::new(vec![content("x"), finished()]),
        ServerConfig::new(1, 4, 16384, 1024)
            .unwrap()
            .with_intake_budget_bytes(16384),
    )
    .await;
    assert_eq!(create(&server, json!({"input":"x"})).await.status, 200);
    // Both the original and converted body must fit the aggregate reservation.
    assert_eq!(
        create(&server, json!({"input":"x".repeat(10000)}))
            .await
            .status,
        429
    );
}

/// A waiting Responses request reserves its expansion allowance, not its
/// declared length, so both its converted input and a stored conversation
/// larger than its body fit.
#[tokio::test]
async fn a_short_follow_up_may_expand_a_larger_stored_conversation_while_waiting() {
    let server = start(
        ScriptedBackendV1::new(vec![content("x"), finished()]),
        stored_config(2).with_unbounded_queue(),
    )
    .await;
    let first = create(&server, json!({"input":"x".repeat(40000)})).await;
    assert_eq!(first.status, 200, "{}", first.body);
    let first = first.json();
    let id = first["id"].as_str().unwrap();
    let follow_up = create(&server, json!({"input":"next","previous_response_id":id})).await;
    assert_eq!(follow_up.status, 200, "{}", follow_up.body);
}

#[tokio::test]
async fn active_foreground_responses_are_not_retrievable_or_cancellable() {
    let (cancelled, mut observed) = tokio::sync::mpsc::unbounded_channel();
    let server = start(
        Arc::new(WaitingBackend {
            cancelled,
            submissions: AtomicUsize::new(0),
        }),
        stored_config(2),
    )
    .await;
    let socket = TcpStream::connect(server.address).await.unwrap();
    let request = post("/v1/responses", r#"{"input":"x","stream":true}"#);
    socket.writable().await.unwrap();
    assert_eq!(socket.try_write(request.as_bytes()).unwrap(), request.len());
    let id = timeout(Duration::from_secs(5), async {
        let mut seen = Vec::new();
        loop {
            let mut bytes = [0; 4096];
            socket.readable().await.unwrap();
            let n = match socket.try_read(&mut bytes) {
                Ok(n) => n,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(error) => panic!("{error}"),
            };
            assert_ne!(n, 0);
            seen.extend_from_slice(&bytes[..n]);
            for line in String::from_utf8_lossy(&seen).lines() {
                if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(value) = serde_json::from_str::<Value>(data)
                    && value["type"] == "response.created"
                {
                    return value["response"]["id"].as_str().unwrap().to_owned();
                }
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(
        server
            .request(&get(&format!("/v1/responses/{id}")))
            .await
            .status,
        404
    );
    assert_eq!(
        server
            .request(&post(&format!("/v1/responses/{id}/cancel"), "{}"))
            .await
            .status,
        404
    );
    assert_eq!(
        create(&server, json!({"input":"next","previous_response_id":id}))
            .await
            .status,
        404
    );
    drop(socket);
    timeout(Duration::from_secs(5), observed.recv())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn storage_refusal_never_submits_generation() {
    let backend = ScriptedBackendV1::new(vec![content("done"), finished()]);
    let server = start(
        backend.clone(),
        stored_config(2).with_responses_store(2, 10),
    )
    .await;
    assert_eq!(create(&server, json!({"input":"x"})).await.status, 400);
    assert_eq!(backend.submissions.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn active_full_store_never_submits_another_generation() {
    let (cancelled, mut observed) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::new(WaitingBackend {
        cancelled,
        submissions: AtomicUsize::new(0),
    });
    let server = start(backend.clone(), stored_config(1)).await;
    let first = create(&server, json!({"input":"x","background":true})).await;
    assert_eq!(first.status, 200);
    assert_eq!(create(&server, json!({"input":"x"})).await.status, 429);
    assert_eq!(backend.submissions.load(Ordering::Relaxed), 1);
    let id = first.json()["id"].as_str().unwrap().to_owned();
    assert_eq!(
        server
            .request(&post(&format!("/v1/responses/{id}/cancel"), "{}"))
            .await
            .status,
        200
    );
    timeout(Duration::from_secs(5), observed.recv())
        .await
        .unwrap()
        .unwrap();
}

struct RefusingBackend {
    submissions: AtomicUsize,
    pause_first: bool,
}
impl GenerationBackend for RefusingBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }
    fn submit(
        &self,
        _: PreparedRequest,
        _: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        let pause = self.submissions.fetch_add(1, Ordering::Relaxed) == 0 && self.pause_first;
        Box::pin(async move {
            if pause {
                std::future::pending::<()>().await;
            }
            Err(AdmissionError::AtCapacity)
        })
    }
}

#[tokio::test]
async fn admission_refusal_releases_the_store_reservation() {
    let backend = Arc::new(RefusingBackend {
        submissions: AtomicUsize::new(0),
        pause_first: false,
    });
    let server = start(backend.clone(), stored_config(1)).await;
    for _ in 0..2 {
        assert_eq!(create(&server, json!({"input":"x"})).await.status, 429);
    }
    assert_eq!(backend.submissions.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn disconnect_during_admission_releases_the_store_reservation() {
    let backend = Arc::new(RefusingBackend {
        submissions: AtomicUsize::new(0),
        pause_first: true,
    });
    let server = start(backend.clone(), stored_config(1)).await;
    let socket = TcpStream::connect(server.address).await.unwrap();
    let request = post("/v1/responses", r#"{"input":"x"}"#);
    socket.writable().await.unwrap();
    assert_eq!(socket.try_write(request.as_bytes()).unwrap(), request.len());
    timeout(Duration::from_secs(5), async {
        while backend.submissions.load(Ordering::Relaxed) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    drop(socket);
    timeout(Duration::from_secs(5), async {
        loop {
            assert_eq!(create(&server, json!({"input":"next"})).await.status, 429);
            if backend.submissions.load(Ordering::Relaxed) == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn stored_failures_use_responses_error_codes() {
    for (event, code, message) in [
        (
            GenerationEvent::Failed {
                message: "failure".to_owned(),
                code: None,
            },
            "server_error",
            "failure",
        ),
        (
            GenerationEvent::AdmissionFailed(AdmissionError::AtCapacity),
            "rate_limit_exceeded",
            "capacity",
        ),
    ] {
        let server = start(ScriptedBackendV1::new(vec![event]), stored_config(1)).await;
        let initial = create(&server, json!({"input":"x","background":true})).await;
        assert_eq!(initial.status, 200);
        let path = format!("/v1/responses/{}", initial.json()["id"].as_str().unwrap());
        timeout(Duration::from_secs(5), async {
            loop {
                let response = server.request(&get(&path)).await;
                assert_eq!(response.status, 200);
                let value = response.json();
                if value["status"] == "failed" {
                    assert_eq!(value["error"]["code"], code);
                    assert!(
                        value["error"]["message"]
                            .as_str()
                            .unwrap()
                            .contains(message)
                    );
                    assert_eq!(value["error"].as_object().unwrap().len(), 2);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}

#[tokio::test]
async fn malformed_nested_item_types_are_refused_before_submission() {
    let backend = ScriptedBackendV1::new(vec![content("ok"), finished()]);
    let server = TestServerV1::start(backend.clone()).await;
    for kind in [json!(42), json!({}), Value::Null, json!(false)] {
        let response = create(
            &server,
            json!({"input":[{"type":kind,"role":"user","content":"x"}]}),
        )
        .await;
        assert_eq!(response.status, 400, "{}", response.body);
        assert_eq!(response.json()["error"]["param"], "input");
    }
    assert_eq!(backend.submissions.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn unsupported_nested_controls_are_refused_before_submission() {
    let backend = ScriptedBackendV1::new(vec![content("ok"), finished()]);
    let server = TestServerV1::start(backend.clone()).await;
    for mut item in [
        json!({"role":"user","content":"x"}),
        json!({"type":"message","role":"user","content":"x"}),
        json!({"type":"function_call","call_id":"c","name":"fixture.lookup","arguments":"{}"}),
        json!({"type":"function_call_output","call_id":"c","output":"x"}),
        json!({"type":"reasoning","content":[{"type":"reasoning_text","text":"x"}],"summary":[]}),
    ] {
        item["unsupported_control"] = json!(true);
        let response = create(&server, json!({"input":[item]})).await;
        assert_eq!(response.status, 400, "{}", response.body);
        assert!(
            response.json()["error"]["message"]
                .as_str()
                .unwrap()
                .contains("unsupported_control")
        );
    }
    for body in [
        json!({"input":[{"role":"user","content":[{"type":"input_text","text":"x","unsupported_control":true}]}]}),
        json!({"input":"x","tools":[tool()],"tool_choice":{"type":"function","name":"fixture.lookup","unsupported_control":true}}),
        json!({"input":"x","text":{"format":{"type":"text","unsupported_control":true}}}),
        json!({"input":[{"type":"reasoning","content":[{"type":"reasoning_text","text":"x"}],"summary":[{"type":"summary_text","text":"x","unsupported_control":true}]}]}),
    ] {
        assert_eq!(create(&server, body).await.status, 400);
    }
    assert_eq!(backend.submissions.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn returned_reasoning_and_message_items_remain_replayable() {
    let backend = ScriptedBackendV1::new(vec![
        GenerationEvent::Reasoning(ReasoningContent {
            delta: "think".to_owned(),
            generated_token_count: 1,
        }),
        content("answer"),
        GenerationEvent::Finished {
            reason: FinishReason::Stop,
            usage: TokenUsage {
                prompt_tokens: 4,
                completion_tokens: 2,
                cached_tokens: None,
            },
        },
    ]);
    let server = TestServerV1::start(backend.clone()).await;
    let first = create(
        &server,
        json!({"input":[{"role":"user","content":"x"}],"reasoning":{"effort":"high"}}),
    )
    .await;
    assert_eq!(first.status, 200);
    let mut input = first.json()["output"].as_array().unwrap().clone();
    assert_eq!(input.len(), 2);
    input.push(
        json!({"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}),
    );
    let next = create(
        &server,
        json!({"input":input,"reasoning":{"effort":"high"}}),
    )
    .await;
    assert_eq!(next.status, 200, "{}", next.body);
    assert!(
        matches!(&backend.prompts()[1].prompt, PreparedPrompt::RenderedChat(text) if text == "assistant:answer~think|user:next")
    );
}
