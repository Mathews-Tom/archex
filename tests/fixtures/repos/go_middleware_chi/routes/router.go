package routes

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5"
	"example.com/app/middleware"
)

func NewRouter(jwtSecret string) http.Handler {
	r := chi.NewRouter()

	r.Use(middleware.LoggingMiddleware)
	r.Use(middleware.AuthMiddleware(jwtSecret))

	r.Route("/api", func(r chi.Router) {
		r.Get("/me", handleMe)
		r.Get("/users", handleListUsers)
	})

	return r
}

func handleMe(w http.ResponseWriter, r *http.Request) {
	userID := r.Context().Value(middleware.UserIDKey)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"userId": userID,
	})
}

func handleListUsers(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]interface{}{
		"users": []string{},
	})
}
