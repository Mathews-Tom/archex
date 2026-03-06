package com.example.events;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@FunctionalInterface
public interface EventListener<T> {
    void onEvent(T event);
}

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@interface Subscribe {
    String topic() default "";
}
