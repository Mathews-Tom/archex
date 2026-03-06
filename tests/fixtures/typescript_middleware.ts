interface Middleware {
    setNext(handler: Middleware): Middleware;
    handle(request: string): string;
}

class BaseMiddleware implements Middleware {
    private nextHandler: Middleware | null = null;

    setNext(handler: Middleware): Middleware {
        this.nextHandler = handler;
        return handler;
    }

    handle(request: string): string {
        if (this.nextHandler) {
            return this.nextHandler.handle(request);
        }
        return '';
    }
}

class AuthMiddleware extends BaseMiddleware {
    handle(request: string): string {
        if (request === 'auth') {
            return 'authenticated';
        }
        return super.handle(request);
    }
}
