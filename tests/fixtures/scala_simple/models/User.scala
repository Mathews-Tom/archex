package com.example.app.models

/** A registered user of the fixture application. */
case class User(id: Int, name: String, email: String, address: Address)

object User {
  val DEFAULT_ROLE: String = "member"

  def validate(user: User): Boolean = user.email.contains("@")

  private def normalizeEmail(email: String): String = email.trim.toLowerCase
}
