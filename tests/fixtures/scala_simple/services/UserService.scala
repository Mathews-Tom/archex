package com.example.app.services

import scala.util.{Failure, Success, Try}
import com.example.app.contracts.Greeter
import com.example.app.models._

/** Coordinates user registration and greeting for the fixture application. */
class UserService(private val greeter: Greeter) {
  private var registered: List[User] = Nil

  def register(user: User): Try[User] = {
    if (validateEmail(user)) {
      registered = user :: registered
      auditLog(s"registered ${user.name}")
      Success(user)
    } else {
      Failure(new IllegalArgumentException(s"invalid email for ${user.name}"))
    }
  }

  def count: Int = registered.size

  private def validateEmail(user: User): Boolean = user.email.contains("@")

  protected[this] def auditLog(message: String): Unit = println(s"[audit] $message")

  override def toString: String = s"UserService(${registered.size} registered)"

  private object Metrics {
    var lookups: Int = 0

    def recordLookup(): Unit = lookups += 1
  }
}
