package com.example.events;

public class OrderService {

    private final EventBus eventBus;

    public OrderService(EventBus eventBus) {
        this.eventBus = eventBus;
    }

    public record OrderCreated(String orderId, String customerId, double total) {}
    public record OrderCancelled(String orderId, String reason) {}

    public void createOrder(String customerId, double total) {
        String orderId = java.util.UUID.randomUUID().toString();
        // persist order ...
        eventBus.publish(new OrderCreated(orderId, customerId, total));
    }

    public void cancelOrder(String orderId, String reason) {
        // update order status ...
        eventBus.publish(new OrderCancelled(orderId, reason));
    }

    public void registerNotificationListener() {
        eventBus.subscribe(OrderCreated.class, event ->
            System.out.println("Notify customer " + event.customerId() + " about order " + event.orderId())
        );
        eventBus.subscribe(OrderCancelled.class, event ->
            System.out.println("Order " + event.orderId() + " cancelled: " + event.reason())
        );
    }
}
