import Combine
import Foundation

class SubscriptionManager {
    private var cancellables = Set<AnyCancellable>()
    private let authPublisher: AuthPublisher
    private let dataPublisher: DataPublisher

    init(authPublisher: AuthPublisher, dataPublisher: DataPublisher) {
        self.authPublisher = authPublisher
        self.dataPublisher = dataPublisher
    }

    func startObserving() {
        authPublisher.statePublisher
            .removeDuplicates { $0.isAuthenticated == $1.isAuthenticated }
            .sink { [weak self] state in
                if state.isAuthenticated {
                    self?.dataPublisher.fetchItems()
                }
            }
            .store(in: &cancellables)

        dataPublisher.filteredPublisher(minValue: 10.0)
            .retry(3)
            .sink(
                receiveCompletion: { completion in
                    if case .failure(let error) = completion {
                        print("Subscription error: \(error)")
                    }
                },
                receiveValue: { items in
                    print("Received \(items.count) filtered items")
                }
            )
            .store(in: &cancellables)
    }

    func cancelAll() {
        cancellables.removeAll()
    }
}
