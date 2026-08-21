package deepwit.examples.neuralImage

import dimwit.*
import dimwit.{Label, |*|}

trait Height derives Label
trait Width derives Label
type Pixel = Height |*| Width
trait Channel derives Label

trait PixelCoordinate derives Label

trait Hidden1 derives Label
trait Hidden2 derives Label
trait Hidden3 derives Label
