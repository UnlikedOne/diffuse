use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use rand::RngCore;

use diffuse_trust::crypto::{hash, hash_hex, sign, verify};

fn make_key() -> SigningKey {
    let mut secret = [0u8; 32];
    OsRng.fill_bytes(&mut secret);
    SigningKey::from_bytes(&secret)
}

#[test]
fn valid_signature_verifies() {
    let key = make_key();
    let pubkey = key.verifying_key().to_bytes();
    let message = b"peer announcement: node holds slice 0:12";

    let signature = sign(&key, message);
    assert!(verify(&pubkey, message, &signature), "valid signature must verify");
}

#[test]
fn tampered_message_fails() {
    let key = make_key();
    let pubkey = key.verifying_key().to_bytes();
    let message = b"node holds slice 0:12";

    let signature = sign(&key, message);
    let tampered = b"node holds slice 0:24";
    assert!(
        !verify(&pubkey, tampered, &signature),
        "signature must not verify for a modified message"
    );
}

#[test]
fn wrong_key_fails() {
    let key = make_key();
    let attacker = make_key();
    let attacker_pubkey = attacker.verifying_key().to_bytes();
    let message = b"node holds slice 0:12";

    let signature = sign(&key, message);
    assert!(
        !verify(&attacker_pubkey, message, &signature),
        "signature must not verify under a different public key"
    );
}

#[test]
fn garbage_signature_fails() {
    let key = make_key();
    let pubkey = key.verifying_key().to_bytes();
    let message = b"node holds slice 0:12";

    let garbage = vec![0u8; 64];
    assert!(!verify(&pubkey, message, &garbage), "garbage signature must fail");

    let too_short = vec![1u8; 10];
    assert!(!verify(&pubkey, message, &too_short), "malformed signature must fail safely");
}

#[test]
fn hash_is_deterministic() {
    let data = b"model shard weights";
    assert_eq!(hash(data), hash(data), "hash must be deterministic");
    assert_eq!(hash_hex(data).len(), 64, "sha256 hex is 64 chars");

    let other = b"different data";
    assert_ne!(hash(data), hash(other), "different data yields different hash");
}
