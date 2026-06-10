use crate::auth::{authenticate, TokenVerifier};

pub struct RequestContext {
    pub bearer_token: String,
}

pub async fn authorize_request(
    request: RequestContext,
    verifier: &dyn TokenVerifier,
) -> Result<(), &'static str> {
    if authenticate(request.bearer_token, verifier).await {
        Ok(())
    } else {
        Err("unauthorized")
    }
}
