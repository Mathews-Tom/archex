pub trait TokenVerifier {
    fn verify(&self, token: &str) -> bool;
}

pub struct StaticTokenVerifier {
    expected: String,
}

impl StaticTokenVerifier {
    pub fn new(expected: String) -> Self {
        Self { expected }
    }
}

impl TokenVerifier for StaticTokenVerifier {
    fn verify(&self, token: &str) -> bool {
        token == self.expected
    }
}

pub async fn authenticate(token: String, verifier: &dyn TokenVerifier) -> bool {
    verifier.verify(&token)
}
