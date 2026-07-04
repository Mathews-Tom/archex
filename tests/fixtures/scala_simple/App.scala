package com.example.app

import scala.util.{Failure, Success}
import com.example.app.contracts.{FriendlyGreeter, Greeter => Greetable}
import com.example.app.models.{Address, User}
import com.example.app.services.UserService
import com.example.app.shapes.{Circle, Empty, Shape, Square}

/** Entry point wiring the fixture application's models, services, and shapes together. */
object Main extends App {
  val greeter: Greetable = new FriendlyGreeter
  val service = new UserService(greeter)
  val address = Address("221B Baker Street", "London")
  val user = User(1, "Ada Lovelace", "ada@example.com", address)

  service.register(user) match {
    case Success(registered) => println(greeter.greet(registered.name))
    case Failure(err) => println(s"registration failed: ${err.getMessage}")
  }

  val shapes: List[Shape] = List(Circle(2.0), Square(3.0), Empty)
  val totalArea: Double = Shape.totalArea(shapes)
  println(s"total area: $totalArea")
}
