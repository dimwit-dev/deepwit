val scala3Version = "3.8.1"

fork := true

lazy val root = project
  .in(file("."))
  .settings(
    name := "deepwit-core",
    organization := "deepwit",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := scala3Version,
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "org.scalatest" %% "scalatest" % "3.2.19" % Test,
      "org.scalacheck" %% "scalacheck" % "1.18.0" % Test,
      "org.scalatestplus" %% "scalacheck-1-18" % "3.2.19.0" % Test,
      "ch.contrafactus" %% "dimwit-core" % "0.1.0-SNAPSHOT" changing (),
      "ch.contrafactus" %% "dimwit-core" % "0.1.0-SNAPSHOT" changing ()
    ),
    // Publishing settings aligned with dimwit core
    Compile / packageSrc / publishArtifact := true,
    Compile / packageDoc / publishArtifact := true
  )
