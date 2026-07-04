package com.example.app.contracts

/** Default friendly implementation of [[Greeter]]. */
trait FriendlyGreeter extends Greeter {
  def greet(name: String): String = s"Hello, $name!"
}

/** Self-type demonstration: only mixable into a concrete [[Greeter]]. */
trait LoudGreeter { self: Greeter =>
  def shout(name: String): String = greet(name).toUpperCase
}
