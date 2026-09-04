import ai.kien.python.Python
import scala.sys.process._

run / fork := true
Global / cancelable := true

ThisBuild / version := "0.1.0"
ThisBuild / scalaVersion := "3.8.1"
ThisBuild / organization := "ch.contrafactus"
ThisBuild / versionScheme := Some("early-semver")

// Publishing to Sonatype Central. The `ch.contrafactus` namespace is verified once for the whole
// organisation, so deepwit needs no verification of its own.
ThisBuild / sonatypeCredentialHost := "central.sonatype.com"
ThisBuild / publishTo := {
  if (isSnapshot.value)
    Some("central-snapshots" at "https://central.sonatype.com/repository/maven-snapshots/")
  else
    sonatypePublishToBundle.value
}
ThisBuild / publishMavenStyle := true
ThisBuild / homepage := Some(url("https://github.com/dimwit-dev/deepwit"))
ThisBuild / licenses := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/dimwit-dev/deepwit"),
    "scm:git@github.com:dimwit-dev/deepwit.git"
  )
)
ThisBuild / developers := List(
  Developer(
    id = "dimwit-dev",
    name = "DeepWit Contributors",
    email = "",
    url = url("https://github.com/dimwit-dev")
  )
)

// scalafix's RemoveUnused reads the compiler's own unused warnings on Scala 3, so it needs both
// semanticdb and -Wunused to be on. Run it with `sbt scalafixAll`.
ThisBuild / semanticdbEnabled := true
ThisBuild / scalacOptions += "-Wunused:imports"

// `core` depends only on released artifacts so that it can be published and consumed without any
// local publishing. `examples` additionally needs plotwit, which is not published anywhere yet and
// so still resolves from the local ivy repo via `publishLocal` in the plotwit checkout.
ThisBuild / resolvers += "Central Portal Snapshots" at "https://central.sonatype.com/repository/maven-snapshots/"

// Consequence of that split: `core` asks for the released dimwit 0.1.0 while the locally published
// plotwit asks for 0.2-SNAPSHOT, which sbt reads as a binary-incompatible conflict under
// early-semver. Let the newer snapshot win in `examples`; `core` on its own resolves 0.1.0.
ThisBuild / libraryDependencySchemes += "ch.contrafactus" %% "dimwit-core" % "always"

addCommandAlias("testAndCoverage", "; clean; coverage; test; coverageReport")

// Publishes `core` only. `examples` depends on plotwit, which is not published.
addCommandAlias("sonaUploadCore", "; project core; sonatypeCentralUpload; project root")

lazy val uvPython: String =
  sys.env.getOrElse(
    "DIMWIT_PYTHON_PATH",
    Seq("uv", "run", "--no-sync", "python", "-c", "import sys; print(sys.executable)").!!.trim
  )
lazy val python = Python(uvPython)
lazy val scalapyJavaOptions = python.scalapyProperties.get.map { case (k, v) => s"-D$k=$v" }.toSeq

lazy val root = (project in file("."))
  .aggregate(core, examples)
  .settings(
    name := "deepwit-root",
    publish / skip := true,
    publishLocal / skip := true,
    publishArtifact := false
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
      "ch.contrafactus" %% "dimwit-core" % "0.1.0"
    ),
    // ScalaPy drives a single embedded CPython interpreter, and two suites importing jax at the same
    // time race into a partially initialized module. Whichever suites happen to touch a tensor first
    // then fail with a circular ImportError, so the suites run one after another.
    Test / parallelExecution := false,
    // Fork with the interpreter that `uvPython` resolved, so a run does not depend on
    // SCALAPY_PYTHON_LIBRARY / SCALAPY_PYTHON_PROGRAMNAME being exported by the shell.
    fork := true,
    javaOptions ++= scalapyJavaOptions,
    description := "A theory-aligned deep learning library for Scala 3, built on DimWit",
    Compile / packageSrc / publishArtifact := true,
    Compile / packageDoc / publishArtifact := true,
    // Ship the library's own sources and docs, not the test ones.
    Test / packageSrc / publishArtifact := false,
    Test / packageDoc / publishArtifact := false
  )

// Examples subproject
lazy val examples = (project in file("examples"))
  .dependsOn(core)
  .settings(
    name := "deepwit-examples",
    libraryDependencies ++= Seq(
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "ch.contrafactus" %% "plotwit-core" % "0.2-SNAPSHOT" changing ()
    ),
    fork := true,
    javaOptions ++= scalapyJavaOptions,
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

// Processes files in /mdocs that need to be copied to the root (e.g. README.md)
lazy val docsRoot = (project in file(".deepwit-docs-root"))
  .enablePlugins(MdocPlugin)
  .dependsOn(core)
  .settings(
    name := "deepwit-docs-root",
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
