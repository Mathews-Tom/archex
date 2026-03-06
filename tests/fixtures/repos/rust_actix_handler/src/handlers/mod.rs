pub mod auth;

use actix_web::web;

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .route("/login", web::post().to(auth::login))
            .route("/me", web::get().to(auth::get_current_user))
            .route("/logout", web::post().to(auth::logout)),
    );
}
