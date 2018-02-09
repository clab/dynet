lazy val root = (project in file("."))
    .settings(
      name         := "dynet_scala_helpers",
      organization := "edu.cmu.dynet",
      version      := "0.0.1-SNAPSHOT"
    )

val DEFAULT_BUILD_PATH = "../../build/contrib/swig"

// The default scala version to use if none was specified from
// outside.  When building with cmake, the scalaversion property
// should always be set; this is only a fallback for other cases.
val DEFAULT_SCALA_VERSION = "2.11.11"

scalaVersion := { sys.props.get("scalaversion") match {
    case Some(p) => p
    case None => {
      println(s"using default scala version ${DEFAULT_SCALA_VERSION}")
      DEFAULT_SCALA_VERSION
    }
}}



// This is where `make` does all its work, and it's where we'll do all our work as well.

lazy val buildPath = settingKey[String]("Build Path")

buildPath := {
  val bp = sys.props.get("buildpath") match {
    case Some(p) => p
    case None => {
      println(s"using default buildpath ${DEFAULT_BUILD_PATH}")
      DEFAULT_BUILD_PATH
    }
  }
  if (new File(bp).exists) {
    bp
  } else {
    throw new IllegalArgumentException(s"buildpath ${bp} does not exist!")
  }
}

lazy val uberjarPath = settingKey[String]("complete path of the uber JAR")

uberjarPath := s"${buildPath.value}/dynet_swigJNI_scala_${scalaBinaryVersion.value}.jar"

excludeFilter in unmanagedJars := "dynet_swigJNI_scala_${scalaBinaryVersion.value}.jar"
excludeFilter in unmanagedSources := HiddenFileFilter || "*.dylib" || "*.so"

// Look for the dynet_swig jar file there.
unmanagedBase  := file( buildPath.value ).getAbsoluteFile

// Put all of the sbt generated classes there.
target := file(s"${buildPath.value}/target/")

// Put the uberjar there.
assemblyOutputPath in assembly := file(uberjarPath.value).getAbsoluteFile

fork := true

val removeUberjar = taskKey[Unit]("Remove Uberjar")

removeUberjar := {
  val uberjar = new java.io.File(uberjarPath.value)
  if (uberjar.exists()) {
    println("removing uberjar")
    uberjar.delete()
  } else {
    println("nothing to remove")
  }
}

assembly := {
  removeUberjar.value
  assembly.value
}

clean := {
  removeUberjar.value
  clean.value
}

assemblyMergeStrategy in assembly := {
  case "libdynet_swig.jnilib" => MergeStrategy.discard
  case "libdynet_swig.so"     => MergeStrategy.discard
  case example if example.contains("/examples/") => MergeStrategy.discard
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

// Don't include Scala libraries in the jar
// see https://github.com/sbt/sbt-assembly/issues/3
// and http://stackoverflow.com/questions/15856739/assembling-a-jar-containing-only-the-provided-dependencies
assembleArtifact in packageScala := false

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"
