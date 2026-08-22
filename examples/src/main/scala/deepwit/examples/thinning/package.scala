package deepwit.examples.thinning

import dimwit.Label

trait Hidden1 derives Label
trait Hidden2 derives Label

trait Row derives Label
trait Column derives Label

/** The colour channels a field is rendered through. */
trait Channel derives Label

/** One thinned parameter set: a single member of the ensemble thinning turns the model into. */
trait Draw derives Label
