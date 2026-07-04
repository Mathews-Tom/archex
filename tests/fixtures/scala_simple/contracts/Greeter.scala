package com.example.app.contracts

/** Abstract greeting contract implemented by concrete greeters. */
trait Greeter {
  def greet(name: String): String
}
