use diffuse_trust::transport::{decrypt, encrypt, KeyExchange};

#[test]
fn two_nodes_derive_same_secret_without_transmitting_it() {
    let alice = KeyExchange::generate();
    let bob = KeyExchange::generate();

    let alice_secret = alice.shared_secret(&bob.public_bytes());
    let bob_secret = bob.shared_secret(&alice.public_bytes());

    assert_eq!(
        alice_secret, bob_secret,
        "both nodes must derive the identical shared secret"
    );
}

#[test]
fn third_party_cannot_derive_the_secret() {
    let alice = KeyExchange::generate();
    let bob = KeyExchange::generate();
    let eve = KeyExchange::generate();

    let real_secret = alice.shared_secret(&bob.public_bytes());
    let eve_attempt = eve.shared_secret(&bob.public_bytes());

    assert_ne!(
        real_secret, eve_attempt,
        "an eavesdropper with only public keys cannot derive the shared secret"
    );
}

#[test]
fn message_encrypted_by_one_is_readable_by_the_other() {
    let alice = KeyExchange::generate();
    let bob = KeyExchange::generate();

    let alice_secret = alice.shared_secret(&bob.public_bytes());
    let bob_secret = bob.shared_secret(&alice.public_bytes());

    let plaintext = b"activation tensor bytes for slice 12:24";
    let ciphertext = encrypt(&alice_secret, plaintext).expect("encrypt");

    assert_ne!(&ciphertext[..], &plaintext[..], "wire data must be encrypted");

    let recovered = decrypt(&bob_secret, &ciphertext).expect("decrypt");
    assert_eq!(recovered, plaintext, "the peer must recover the exact plaintext");
}

#[test]
fn eavesdropper_cannot_decrypt() {
    let alice = KeyExchange::generate();
    let bob = KeyExchange::generate();
    let eve = KeyExchange::generate();

    let alice_secret = alice.shared_secret(&bob.public_bytes());
    let eve_secret = eve.shared_secret(&bob.public_bytes());

    let ciphertext = encrypt(&alice_secret, b"private prompt activations").expect("encrypt");

    let result = decrypt(&eve_secret, &ciphertext);
    assert!(
        result.is_err(),
        "an eavesdropper with the wrong secret must fail to decrypt"
    );
}

#[test]
fn tampered_ciphertext_is_rejected() {
    let alice = KeyExchange::generate();
    let bob = KeyExchange::generate();

    let secret = alice.shared_secret(&bob.public_bytes());
    let mut ciphertext = encrypt(&secret, b"integrity matters").expect("encrypt");

    let last = ciphertext.len() - 1;
    ciphertext[last] ^= 0xff;

    let result = decrypt(&secret, &ciphertext);
    assert!(
        result.is_err(),
        "AEAD must detect tampering and refuse to decrypt"
    );
}
