package deepwit.examples

/** The run a training main wrote most recently under `root`.
  *
  * Runs are named for the hour they were written, so the newest of them is the last in order.
  */
def newestRun(root: String): String =
  val runs = Option(java.io.File(root).listFiles()).getOrElse(Array.empty[java.io.File]).filter(_.isDirectory)
  require(runs.nonEmpty, s"Nothing to load: $root holds no run yet. Train first.")
  runs.map(_.getPath).max
