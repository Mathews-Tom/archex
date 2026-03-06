use actix_web::dev::{Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::Error;
use futures::future::{ok, LocalBoxFuture, Ready};
use std::task::{Context, Poll};

pub struct AuthGuard;

impl<S, B> Transform<S, ServiceRequest> for AuthGuard
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthGuardMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ok(AuthGuardMiddleware { service })
    }
}

pub struct AuthGuardMiddleware<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for AuthGuardMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let has_auth = req.headers().get("Authorization").is_some();
        if !has_auth {
            return Box::pin(async {
                Err(actix_web::error::ErrorUnauthorized("missing token"))
            });
        }
        let fut = self.service.call(req);
        Box::pin(async move { fut.await })
    }
}
