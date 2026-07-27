use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use rand::RngCore;

use diffuse_trust::transport::KeyExchange;

pub struct Identity {
    pub signing_key: SigningKey,
    pub key_exchange: KeyExchange,
}

impl Identity {
    pub fn generate() -> Self {
        let mut secret = [0u8; 32];
        OsRng.fill_bytes(&mut secret);
        let signing_key = SigningKey::from_bytes(&secret);
        let key_exchange = KeyExchange::generate();
        Self {
            signing_key,
            key_exchange,
        }
    }

    pub fn public_hex(&self) -> String {
        hex::encode(self.signing_key.verifying_key().to_bytes())
    }

    pub fn short_id(&self) -> String {
        let full = self.public_hex();
        full[..16].to_string()
    }

    pub fn signing_public(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    pub fn kx_public(&self) -> [u8; 32] {
        self.key_exchange.public_bytes()
    }
}
