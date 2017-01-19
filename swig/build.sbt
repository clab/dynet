
name := "dynet_scala_helpers"

scalaVersion := "2.11.8"

val DEFAULT_BUILD_PATH = "../build/swig/javajar"

// This is where `make` does all its work, and it's where we'll do all our work as well.
val buildPath = {
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

// Look for the dynet_swig jar file there.
unmanagedBase := file( buildPath ).getAbsoluteFile

// Put all of the sbt generated classes there.
target := file(s"${buildPath}/target/")

// Put the uberjar there.
assemblyOutputPath in assembly := file(s"${buildPath}/../scalajar/dynet_swigJNI_scala.jar")
    .getAbsoluteFile

fork := true

// Don't include Scala libraries in the jar
// see https://github.com/sbt/sbt-assembly/issues/3
// and http://stackoverflow.com/questions/15856739/assembling-a-jar-containing-only-the-provided-dependencies
assembleArtifact in packageScala := false

// And look there for java libraries when running.
javaOptions += s"-Djava.library.path=${buildPath}"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"
