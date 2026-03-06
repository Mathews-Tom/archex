import Combine
import Foundation

struct AuthState {
    let isAuthenticated: Bool
    let userId: String?
    let token: String?
}

class AuthPublisher {
    private let authSubject = CurrentValueSubject<AuthState, Never>(
        AuthState(isAuthenticated: false, userId: nil, token: nil)
    )

    var statePublisher: AnyPublisher<AuthState, Never> {
        authSubject.eraseToAnyPublisher()
    }

    var isAuthenticated: Bool {
        authSubject.value.isAuthenticated
    }

    func login(token: String) {
        let userId = decodeUserId(from: token)
        authSubject.send(AuthState(isAuthenticated: true, userId: userId, token: token))
    }

    func logout() {
        authSubject.send(AuthState(isAuthenticated: false, userId: nil, token: nil))
    }

    private func decodeUserId(from token: String) -> String {
        let parts = token.split(separator: ".")
        guard parts.count >= 2,
              let data = Data(base64Encoded: String(parts[1])),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sub = json["sub"] as? String else {
            return "unknown"
        }
        return sub
    }
}
