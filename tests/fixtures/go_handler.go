package handler

type Handler interface {
	SetNext(handler Handler) Handler
	Handle(request string) string
}

type BaseHandler struct {
	next Handler
}

func (h *BaseHandler) SetNext(handler Handler) Handler {
	h.next = handler
	return handler
}

func (h *BaseHandler) Handle(request string) string {
	if h.next != nil {
		return h.next.Handle(request)
	}
	return ""
}

type AuthHandler struct {
	BaseHandler
}

func (h *AuthHandler) Handle(request string) string {
	if request == "auth" {
		return "authenticated"
	}
	return h.BaseHandler.Handle(request)
}
