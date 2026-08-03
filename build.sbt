run / fork := true
Global / cancelable := true

ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.8.1"
ThisBuild / organization := "ch.contrafactus"

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
    javaOptions ++= {
      if (sys.props("os.name").toLowerCase.contains("mac")) {
        Seq("-XstartOnFirstThread") // For MacOS to run Python with GUI support
      } else {
        Seq.empty
      }
    }
  )
