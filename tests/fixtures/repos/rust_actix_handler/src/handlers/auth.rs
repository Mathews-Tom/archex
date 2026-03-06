use actix_web::{web, HttpRequest, HttpResponse};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: usize,
    pub roles: Vec<String>,
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

pub async fn login(body: web::Json<LoginRequest>) -> HttpResponse {
    let claims = Claims {
        sub: body.email.clone(),
        exp: (chrono::Utc::now() + chrono::Duration::hours(24)).timestamp() as usize,
        roles: vec!["user".to_string()],
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(b"secret"),
    )
    .unwrap();

    HttpResponse::Ok().json(serde_json::json!({ "token": token }))
}

pub async fn get_current_user(req: HttpRequest) -> HttpResponse {
    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());

    let token = match auth_header {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        _ => return HttpResponse::Unauthorized().finish(),
    };

    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(b"secret"),
        &Validation::default(),
    );

    match token_data {
        Ok(data) => HttpResponse::Ok().json(serde_json::json!({
            "userId": data.claims.sub,
            "roles": data.claims.roles,
        })),
        Err(_) => HttpResponse::Unauthorized().finish(),
    }
}

pub async fn logout() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({ "message": "logged out" }))
}
