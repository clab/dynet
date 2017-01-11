
name := "dynet_scala_helpers"

scalaVersion := "2.11.8"

// This is where `make` does all its work, and it's where we'll do all our work as well.
val buildPath = "../build/swig"

// Look for the dynet_swig jar file there.
unmanagedBase := file( buildPath ).getAbsoluteFile

// Put all of the sbt generated classes there.
target := file(s"${buildPath}/target/")

// Put the uberjar there.
assemblyOutputPath in assembly := file(s"${buildPath}/dynet_swigJNI_scala.jar").getAbsoluteFile

fork in run := true

// And look there for java libraries when running.
javaOptions in run += s"-Djava.library.path=${buildPath}"
