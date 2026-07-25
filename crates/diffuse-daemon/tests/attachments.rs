use diffuse_daemon::commands::collect_attachments;

fn write(name: &str, bytes: &[u8]) -> String {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, bytes).expect("write fixture");
    path.to_string_lossy().into_owned()
}

#[test]
fn kind_is_taken_from_the_extension_for_universal_attachments() {
    let png = write("diffuse-att.png", b"\x89PNG fake");
    let wav = write("diffuse-att.wav", b"RIFF fake");
    let mp4 = write("diffuse-att.mp4", b"ftyp fake");

    let found = collect_attachments(&[], &[], &[], &[png, wav, mp4]).expect("collect");
    let kinds: Vec<&str> = found.iter().map(|a| a.kind.as_str()).collect();
    assert_eq!(kinds, vec!["image", "audio", "video"]);
    assert_eq!(found[0].mime, "image/png");
    assert_eq!(found[1].mime, "audio/wav");
}

#[test]
fn an_explicit_flag_overrides_an_unhelpful_extension() {
    let odd = write("diffuse-att.bin", b"bytes");
    let found = collect_attachments(&[odd], &[], &[], &[]).expect("collect");
    assert_eq!(found[0].kind, "image");
}

#[test]
fn an_unknown_extension_asks_which_kind_it_is() {
    let odd = write("diffuse-att.xyz", b"bytes");
    let err = collect_attachments(&[], &[], &[], &[odd]).unwrap_err().to_string();
    assert!(err.contains("--image"), "the error must say how to disambiguate: {}", err);
}

#[test]
fn a_missing_or_empty_attachment_is_refused_before_the_network() {
    let missing = std::env::temp_dir().join("diffuse-att-missing.png");
    let err = collect_attachments(&[missing.to_string_lossy().into_owned()], &[], &[], &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot read"), "{}", err);

    let empty = write("diffuse-att-empty.png", b"");
    let err = collect_attachments(&[empty], &[], &[], &[]).unwrap_err().to_string();
    assert!(err.contains("empty"), "{}", err);
}

#[test]
fn attachments_keep_their_order_across_kinds() {
    let a = write("diffuse-order-a.png", b"a");
    let b = write("diffuse-order-b.wav", b"b");
    let found = collect_attachments(&[a], &[b], &[], &[]).expect("collect");
    assert_eq!(found[0].label, "diffuse-order-a.png");
    assert_eq!(found[1].label, "diffuse-order-b.wav");
}
