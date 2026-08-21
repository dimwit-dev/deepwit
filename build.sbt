import ai.kien.python.Python
import scala.sys.process._

run / fork := true
Global / cancelable := true

ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.8.1"
ThisBuild / organization := "ch.contrafactus"

// scalafix's RemoveUnused reads the compiler's own unused warnings on Scala 3, so it needs both
// semanticdb and -Wunused to be on. Run it with `sbt scalafixAll`.
ThisBuild / semanticdbEnabled := true
ThisBuild / scalacOptions += "-Wunused:imports"

// Add resolver for snapshot dependencies
ThisBuild / resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

addCommandAlias("testAndCoverage", "; clean; coverage; test; coverageReport")

lazy val root = (project in file("."))
  .aggregate(core, examples)
  .settings(
    name := "deepwit-root"
  )

lazy val core = (project in file("core"))
  .settings(
    name := "deepwit-core",
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "org.scalatest" %% "scalatest" % "3.2.19" % Test,
      "org.scalacheck" %% "scalacheck" % "1.18.0" % Test,
      "org.scalatestplus" %% "scalacheck-1-18" % "3.2.19.0" % Test,
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "ch.contrafactus" %% "dimwit-core" % "0.1.0-SNAPSHOT" changing ()
    ),
    // ScalaPy drives a single embedded CPython interpreter, and two suites importing jax at the same
    // time race into a partially initialized module. Whichever suites happen to touch a tensor first
    // then fail with a circular ImportError, so the suites run one after another.
    Test / parallelExecution := false,
    Compile / packageSrc / publishArtifact := true,
    Compile / packageDoc / publishArtifact := true
  )

// Examples subproject
lazy val examples = (project in file("examples"))
  .dependsOn(core)
  .settings(
    name := "deepwit-examples",
    libraryDependencies ++= Seq(
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "ch.contrafactus" %% "plotwit-core" % "0.1.0-SNAPSHOT" changing ()
    ),
    fork := true,
    // Don't publish examples
    publish := {},
    publishLocal := {},
    publishArtifact := false,
    // Examples source directory
    Compile / scalaSource := baseDirectory.value,
    Compile / resourceDirectory := baseDirectory.value / "src" / "main" / "resources",
    javaOptions ++= Seq(
      // "-XX:G1PeriodicGCInterval=1000"
      "-XX:+UseZGC",
      "-XX:ZCollectionInterval=1" // Forces a GC cycle every 1 second, regardless of heap usage
    )
  )

lazy val uvPython: String =
  sys.env.getOrElse(
    "DIMWIT_PYTHON_PATH",
    Seq("uv", "run", "--no-sync", "python", "-c", "import sys; print(sys.executable)").!!.trim
  )
lazy val python = Python(uvPython)
lazy val scalapyJavaOptions = python.scalapyProperties.get.map { case (k, v) => s"-D$k=$v" }.toSeq

// Processes files in /mdocs that need to be copied to the root (e.g. README.md)
lazy val docsRoot = (project in file(".dimwit-docs-root"))
  .enablePlugins(MdocPlugin)
  .dependsOn(core)
  .settings(
    name := "dimwit-docs-root",
    publish / skip := true,
    mdocIn := (ThisBuild / baseDirectory).value / "mdocs",
    mdocOut := (ThisBuild / baseDirectory).value,
    mdocExtraArguments := Seq("--no-link-hygiene"),
    mdocVariables := Map(
      "VERSION" -> version.value
    ),
    fork := true,
    javaOptions ++= scalapyJavaOptions,
    envVars := (ThisBuild / envVars).value
  )
