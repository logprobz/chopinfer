# Served native MoE tuning

Flash TP4 native prefill uses gate/up r2w6s3, down r1w4s2 and
`Tune::Scanned`, as declared by `moe_native_entries`. Locked lab recipes
retain earlier entries as equality controls for the served checks in
`tests/sm120/native_moe_prefill`.

On 2026-10-21, a two-device pair at 250 W compared balanced, skewed and
captured 8192-row inputs. Every bitwise arm matched the served bytes on all
four cases. The lean gate/up loop improved the paired median by 5.0%; the
paired lean down tile improved it by 4.3%, or 6.1% on captured cases. Every
round agreed with its median ratio within 0.03. The two-CTA gate/up tile did
not improve the paired median. These kernel measurements explain the
adopted lineage; they are not whole-serving performance evidence.
