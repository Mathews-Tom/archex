package com.example.app.util

/** Small string helpers used across the fixture application. */
object StringUtils {
  val MAX_LENGTH: Int = 255

  def slugify(input: String): String = normalize(input).replaceAll("\\s+", "-")

  def truncate(input: String, limit: Int): String =
    if (input.length <= limit) input else input.substring(0, limit)

  private def normalize(input: String): String = input.trim.toLowerCase

  class Builder {
    private var parts: List[String] = Nil

    def append(part: String): Builder = {
      parts = part :: parts
      this
    }

    def build(): String = parts.reverse.mkString(" ")
  }
}
