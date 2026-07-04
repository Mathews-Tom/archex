package com.example.app.shapes

/** A sealed shape hierarchy exercising pattern matching over case classes. */
sealed trait Shape

case class Circle(radius: Double) extends Shape
case class Square(side: Double) extends Shape
case object Empty extends Shape

object Shape {
  def area(shape: Shape): Double = shape match {
    case Circle(radius) => math.Pi * radius * radius
    case Square(side) => side * side
    case Empty => 0.0
  }

  def totalArea(shapes: List[Shape]): Double = {
    val areas = for {
      shape <- shapes
      area = Shape.area(shape)
      if area >= 0.0
    } yield area
    areas.sum
  }
}
