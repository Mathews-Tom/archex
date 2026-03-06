import Combine
import Foundation

struct DataItem: Identifiable {
    let id: UUID
    let name: String
    let value: Double
}

class DataPublisher {
    private let itemsSubject = PassthroughSubject<[DataItem], Error>()
    private var cancellables = Set<AnyCancellable>()

    var itemsPublisher: AnyPublisher<[DataItem], Error> {
        itemsSubject.eraseToAnyPublisher()
    }

    func fetchItems() {
        URLSession.shared.dataTaskPublisher(for: URL(string: "https://api.example.com/items")!)
            .map(\.data)
            .decode(type: [DataItem].self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    if case .failure(let error) = completion {
                        self?.itemsSubject.send(completion: .failure(error))
                    }
                },
                receiveValue: { [weak self] items in
                    self?.itemsSubject.send(items)
                }
            )
            .store(in: &cancellables)
    }

    func filteredPublisher(minValue: Double) -> AnyPublisher<[DataItem], Error> {
        itemsPublisher
            .map { items in items.filter { $0.value >= minValue } }
            .eraseToAnyPublisher()
    }
}
